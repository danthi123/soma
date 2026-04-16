"""GCS-backed implementation of :class:`soma.storage.ObjectStore`.

Third concrete adapter behind the Protocol landed in Phase 30, sibling
to :class:`soma.storage.s3.S3ObjectStore`. Same contract as
:class:`LocalFSObjectStore` — ``KeyError`` on missing object,
idempotent ``delete``, streaming read/write — but with a Google Cloud
Storage bucket + prefix as the backing store.

Unlocks SOMA bundle storage on Google Cloud Run, Cloud Functions, GKE,
and any deployment where the local disk is ephemeral and GCS is the
durable tier. Credentials follow the standard Google Application
Default Credentials (ADC) chain: env var
``GOOGLE_APPLICATION_CREDENTIALS``, metadata server on GCP, or
``gcloud auth application-default login`` for local dev.

Design notes:

* **Atomicity.** GCS PUT is atomic per object — same as S3, no
  ``.tmp`` staging dance needed. ``list_prefix`` still skips stray
  ``.tmp`` siblings for cross-adapter consistency (a hybrid deploy
  sharing a bucket with a local-writer process could leak these).

* **Credentials.** The google-cloud-storage default ADC chain is the
  only path: env vars, metadata server, gcloud creds. No SOMA-specific
  creds shim, so ops teams can layer their existing Google IAM
  discipline on top. The ``project=`` kwarg is forwarded to the
  ``storage.Client`` constructor for setups that need an explicit
  project override (quota attribution, cross-project service account
  usage).

* **Streaming.** ``blob.upload_from_file`` / ``blob.download_to_file``
  are streaming-by-default so hundred-MB bundle files don't
  materialise in RAM. ``get_stream`` pre-loads into a
  :class:`io.BytesIO` for the same reason the S3 adapter does (see
  below) — torch.load / numpy.load need seek, which a raw GCS
  download stream doesn't provide.

* **Staging temp dir.** :attr:`local_root` lazily materialises a
  local mirror of the GCS prefix so MemoryLayer's WAL replay +
  ``bundle.lock`` portalocker sidecar (which both need a real local
  path) work transparently against remote storage. First access
  downloads every object; :meth:`close` uploads any new / modified
  files back. See :class:`soma.memory.api.MemoryLayer.load` for the
  integration seam. Identical shape to the S3 adapter — tests mirror
  that adapter's ``test_s3_local_root_staging_round_trip``.

* **Missing-object semantics.** ``blob.download_as_bytes`` on an
  absent blob raises ``google.cloud.exceptions.NotFound``; ``blob.delete``
  on a missing blob raises the same. We catch both and translate —
  ``KeyError`` for reads (Protocol contract), silent no-op for deletes
  (S3 idempotency contract). This is the main place GCS diverges
  from S3, where ``delete_object`` on a missing key is naturally a
  no-op returning 204.
"""

from __future__ import annotations

import atexit
import contextlib
import io
import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

try:
    from google.cloud import storage as _gcs
    from google.cloud.exceptions import NotFound as _GCSNotFound

    _HAS_GCS = True
except ImportError:  # pragma: no cover - env-dependent
    _gcs = None  # type: ignore[assignment]
    _GCSNotFound = Exception  # type: ignore[assignment,misc]
    _HAS_GCS = False


logger = logging.getLogger("soma.storage.gcs")


class GCSObjectStore:
    """Store bytes as GCS objects under ``bucket/prefix/``.

    Keys are forward-slash relative paths; the adapter glues
    ``prefix`` on the way in and strips it on the way out so callers
    see the same flat keyspace they'd see on :class:`LocalFSObjectStore`
    or :class:`S3ObjectStore`.

    Parameters
    ----------
    bucket:
        GCS bucket name. Must exist — the adapter never creates the
        bucket (callers' ops path handles provisioning).
    prefix:
        Optional key prefix (``""`` for bucket root). Trailing slash
        is stripped. Two stores pointing at the same bucket with
        different prefixes see disjoint keyspaces.
    project:
        GCS project ID. Optional — if unset, the client picks it up
        from ADC (``gcloud config``, metadata server, etc.). Set
        explicitly for cross-project service-account usage or quota
        attribution.
    client:
        Pre-built ``google.cloud.storage.Client``. When supplied, the
        constructor skips creating its own client — handy for tests
        sharing one emulator and for advanced callers wiring up
        custom credentials, retries, user-agent strings, etc.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        project: str | None = None,
        client: Any = None,
    ) -> None:
        if not _HAS_GCS:
            raise ImportError(
                'google-cloud-storage not installed; '
                'install with: pip install "soma[gcs]"'
            )
        self._bucket_name: str = bucket
        # Normalise: strip leading + trailing slashes and backslashes so
        # ``prefix="/bundle/"`` and ``prefix="bundle"`` are equivalent
        # (callers read ``gs://bucket/prefix/`` and sometimes copy the
        # URL verbatim).
        self._prefix: str = prefix.replace("\\", "/").strip("/")
        if client is not None:
            self._client = client
        else:
            self._client = _gcs.Client(project=project)
        self._bucket = self._client.bucket(bucket)
        # Lazy staging state (see ``local_root``). Initialised to None
        # so the temp dir is only created on first access — callers that
        # only do direct put_bytes / get_bytes (the common Save/Load
        # path) never pay for it.
        self._local_root: Path | None = None
        self._staged_keys: set[str] = set()
        self._closed: bool = False
        # Register the sync-on-close hook so a crash before the caller
        # calls ``close()`` still pushes staged changes back to GCS.
        atexit.register(self._atexit_close)

    # ------------------------------------------------------------------
    # Key / prefix plumbing
    # ------------------------------------------------------------------
    def _full_key(self, key: str) -> str:
        """Turn a caller-supplied key into an absolute GCS object name.

        Normalises backslashes (Windows callers sometimes leak them
        through) and strips any leading slash so the key never
        accidentally escapes the configured prefix.
        """
        normalised = key.replace("\\", "/").lstrip("/")
        if self._prefix:
            return f"{self._prefix}/{normalised}"
        return normalised

    def _strip_prefix(self, full_key: str) -> str:
        """Remove the configured prefix from an absolute GCS object name."""
        if not self._prefix:
            return full_key
        head = f"{self._prefix}/"
        if full_key.startswith(head):
            return full_key[len(head):]
        # Shouldn't happen for keys we ourselves emitted; return as-is
        # rather than silently swallowing a list-prefix bug.
        return full_key

    # ------------------------------------------------------------------
    # Byte-oriented methods
    # ------------------------------------------------------------------
    def get_bytes(self, key: str) -> bytes:
        full = self._full_key(key)
        blob = self._bucket.blob(full)
        try:
            return bytes(blob.download_as_bytes())
        except _GCSNotFound as exc:
            raise KeyError(key) from exc

    def put_bytes(self, key: str, data: bytes) -> None:
        full = self._full_key(key)
        blob = self._bucket.blob(full)
        # ``upload_from_file`` with a BytesIO is the canonical bytes
        # path — ``upload_from_string`` auto-detects str vs bytes
        # but has historically been finicky about content-type
        # sniffing on emulators. BytesIO is the one shape that works
        # everywhere.
        blob.upload_from_file(io.BytesIO(data), rewind=True)

    # ------------------------------------------------------------------
    # Stream-oriented methods — preferred for large payloads
    # ------------------------------------------------------------------
    def get_stream(self, key: str) -> BinaryIO:
        """Return a readable binary stream for ``key``.

        The returned stream is a :class:`io.BytesIO` pre-loaded with
        the full object body — seekable, ``tell()``-able, and usable
        by ``torch.load`` / ``numpy.load`` / ``zipfile.ZipFile`` and
        friends that need to seek around the payload.

        This means the whole body is resident in RAM while the stream
        is open. For very large blobs (hundred-MB+ vectors) callers
        that want true streaming should either:

        * Use :meth:`get_bytes` (explicit: "I want bytes").
        * Use the ``local_root`` staging path — mirrors the GCS
          prefix to a temp dir once, reads files with native
          ``open()``.
        * Call ``blob.download_to_file`` themselves with a custom
          chunked consumer.

        Matches the Phase-31 S3 adapter's ``get_stream`` shape exactly:
        torch's ``PyTorchFileReader`` needs seek + raw C IO, which a
        streaming GCS download doesn't give us, so we slurp then wrap.
        Without this the GCS path would silently fail the first time
        a user loaded a bundle over ``gs://``.
        """
        full = self._full_key(key)
        blob = self._bucket.blob(full)
        try:
            data = blob.download_as_bytes()
        except _GCSNotFound as exc:
            raise KeyError(key) from exc
        return io.BytesIO(bytes(data))

    def put_stream(self, key: str, stream: BinaryIO) -> None:
        full = self._full_key(key)
        blob = self._bucket.blob(full)
        # ``upload_from_file`` consumes the stream in chunks (google's
        # resumable-media default is 8 MB) so hundred-MB payloads
        # don't materialise in RAM.
        blob.upload_from_file(stream, rewind=True)

    # ------------------------------------------------------------------
    # Directory-style helpers
    # ------------------------------------------------------------------
    def list_prefix(self, prefix: str) -> Iterator[str]:
        """Yield keys under the store prefix that start with ``prefix``.

        Uses ``client.list_blobs(bucket, prefix=...)`` which is
        server-side paginated — the client iterates pages
        transparently, so bundles with more than 1000 objects don't
        silently truncate. ``.tmp`` siblings are filtered out for
        parity with the local + S3 adapters.
        """
        normalised = prefix.replace("\\", "/").lstrip("/")
        full_prefix = f"{self._prefix}/{normalised}" if self._prefix else normalised
        # ``list_blobs`` returns an iterator that auto-paginates.
        for blob in self._client.list_blobs(self._bucket, prefix=full_prefix):
            full = blob.name
            if full.endswith(".tmp"):
                continue
            rel = self._strip_prefix(full)
            if rel.startswith(prefix):
                yield rel

    def delete(self, key: str) -> None:
        """Remove ``key``. Idempotent — GCS ``blob.delete`` on a missing
        key raises ``NotFound``, unlike S3 which is naturally a no-op;
        the adapter swallows the exception so callers get the same
        Protocol contract regardless of backend."""
        full = self._full_key(key)
        blob = self._bucket.blob(full)
        try:
            blob.delete()
        except _GCSNotFound:
            return

    def exists(self, key: str) -> bool:
        full = self._full_key(key)
        blob = self._bucket.blob(full)
        # ``blob.exists()`` returns bool without raising on 404 — it
        # issues a HEAD-equivalent metadata request.
        return bool(blob.exists())

    # ------------------------------------------------------------------
    # Local staging — lazy mirror of the GCS prefix as a real directory
    # ------------------------------------------------------------------
    @property
    def local_root(self) -> Path:
        """Lazy local mirror of the GCS prefix for WAL / lock use.

        MemoryLayer's WAL replay and ``bundle.lock`` sidecar both need
        a real on-disk directory (``portalocker`` locks a file,
        ``json.loads`` the WAL header as lines). For remote stores we
        materialise a temp dir on first access and keep it in sync:

        * **On first access:** create a unique temp dir and download
          every object under the store's prefix into it, preserving
          the key hierarchy.
        * **Subsequent access:** return the cached temp dir (no
          re-download — in-flight mutations live in the temp dir).
        * **On :meth:`close`:** walk the temp dir and upload every
          file back to GCS at the matching key.

        The invariant is "whatever is under ``local_root`` at close
        time is what ends up in GCS". Calls that go directly through
        :meth:`put_bytes` / :meth:`get_bytes` bypass the staging dir
        (they talk to GCS directly) — the staging path is only walked
        by code that needs a filesystem handle.

        Identical shape to :attr:`S3ObjectStore.local_root`.
        """
        if self._local_root is None:
            self._local_root = Path(
                tempfile.mkdtemp(prefix="soma-gcs-stage-")
            )
            self._download_into_stage(self._local_root)
        return self._local_root

    def _download_into_stage(self, root: Path) -> None:
        """Mirror every object under the store prefix into ``root``.

        Forward-slash GCS keys map to ``os.sep``-separated directory
        hierarchies under ``root``. On Windows ``Path`` handles both
        directions transparently.
        """
        for rel_key in list(self.list_prefix("")):
            target = root / Path(rel_key)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = self.get_bytes(rel_key)
            except KeyError:
                # Raced with a concurrent delete — skip the vanished key.
                continue
            target.write_bytes(data)

    def close(self) -> None:
        """Push the local staging dir (if any) back to GCS and tear it down.

        Safe to call multiple times. When :attr:`local_root` was never
        accessed this is a no-op — no temp dir was created, nothing to
        sync. Otherwise every file under the stage dir is uploaded
        (overwriting any GCS object at the same key — we don't try to
        detect "unchanged" because the caller explicitly opted into
        the staging path when they touched :attr:`local_root`).
        """
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):  # pragma: no cover - py<3.13 quirk
            atexit.unregister(self._atexit_close)
        if self._local_root is None:
            return
        root = self._local_root
        try:
            self._upload_from_stage(root)
        finally:
            # Best-effort cleanup — a lingering temp dir is harmless
            # (the OS reaps it on reboot) and we'd rather eat the
            # cleanup error than mask an upload failure.
            import shutil

            try:
                shutil.rmtree(root, ignore_errors=True)
            except Exception:  # pragma: no cover - defensive only
                logger.warning("failed to clean up GCS staging dir %s", root)

    def _upload_from_stage(self, root: Path) -> None:
        """Upload every file under ``root`` back to GCS at the matching key.

        Uses ``put_bytes`` (read+upload) rather than a threaded
        transfer manager because this path also runs from an
        ``atexit`` hook and google's background threads refuse to
        schedule new work after interpreter shutdown. Staging files
        are typically small (WAL records, bundle.lock, index.json);
        the large-blob path goes through :meth:`put_bytes` /
        :meth:`put_stream` directly, not the stage dir.
        """
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                # Skip ``.tmp`` siblings: the local adapter's atomic
                # write lands a .tmp, then ``os.replace``s it — if we
                # ever stage via the local adapter path we must not
                # push the transient to GCS.
                if fname.endswith(".tmp"):
                    continue
                abs_path = Path(dirpath) / fname
                rel = abs_path.relative_to(root).as_posix()
                self.put_bytes(rel, abs_path.read_bytes())

    def _atexit_close(self) -> None:
        """atexit hook so a caller who forgot to call close() still flushes.

        Swallows every exception silently. By the time atexit runs the
        logger / stderr may already be closed, and we don't want our
        best-effort sync to foul the interpreter-shutdown path. In
        tests the emulator is torn down before atexit fires and the
        client fails the upload — harmless.
        """
        with contextlib.suppress(Exception):  # pragma: no cover - best-effort
            self.close()
