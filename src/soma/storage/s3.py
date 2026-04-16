"""S3-backed implementation of :class:`soma.storage.ObjectStore`.

Second concrete adapter behind the Protocol landed in Phase 30. Same
contract as :class:`LocalFSObjectStore` — ``KeyError`` on missing
object, idempotent ``delete``, streaming-first ``get_stream`` /
``put_stream`` — but with an S3 bucket + prefix as the backing store.

Unlocks SOMA bundle storage on any scale-to-zero platform where local
disk is ephemeral: AWS Lambda, Cloud Run, Fly Machines, as well as
S3-compatible object stores (MinIO, Cloudflare R2, DigitalOcean
Spaces) via the ``endpoint_url=`` kwarg.

Design notes:

* **Atomicity.** S3 PUT is atomic per key — no ``.tmp`` staging
  file needed (unlike local FS where writes are visible before the
  rename). ``list_prefix`` still skips stray ``.tmp`` siblings so a
  crashed local writer in a hybrid deploy doesn't leak half-files
  into S3 listings.

* **Credentials.** Boto3's default chain is the only path: env vars,
  ``~/.aws/credentials``, IAM role, SSO. No SOMA-specific creds shim,
  so ops teams can layer their existing AWS discipline on top.

* **Streaming.** ``upload_fileobj`` / ``get_object`` + ``StreamingBody``
  are used so hundred-MB bundle files don't materialise in RAM.
  ``StreamingBody`` is not seekable; callers that need ``seek`` (e.g.
  ``numpy.load`` on a compressed ``.npy``) should buffer into a
  ``BytesIO`` themselves — the Protocol contract leaves this up to
  the caller.

* **Staging temp dir.** :attr:`local_root` lazily materialises a
  local mirror of the S3 prefix so MemoryLayer's WAL replay +
  ``bundle.lock`` portalocker sidecar (which both need a real local
  path) work transparently against remote storage. First access
  downloads every object; :meth:`close` uploads any new / modified
  files back. See :class:`soma.memory.api.MemoryLayer.load` for the
  integration seam.
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
    import boto3
    from botocore.exceptions import ClientError

    _HAS_BOTO3 = True
except ImportError:  # pragma: no cover - env-dependent
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[assignment,misc]
    _HAS_BOTO3 = False


logger = logging.getLogger("soma.storage.s3")


class S3ObjectStore:
    """Store bytes as S3 objects under ``bucket/prefix/``.

    Keys are forward-slash relative paths; the adapter glues
    ``prefix`` on the way in and strips it on the way out so callers
    see the same flat keyspace they'd see on :class:`LocalFSObjectStore`.

    Parameters
    ----------
    bucket:
        S3 bucket name. Must exist — the adapter never creates the
        bucket (callers' ops path handles provisioning).
    prefix:
        Optional key prefix (``""`` for bucket root). Trailing slash
        is stripped. Two stores pointing at the same bucket with
        different prefixes see disjoint keyspaces.
    endpoint_url:
        Override the default AWS endpoint. Set to ``http://localhost:9000``
        for MinIO, your R2 account URL for Cloudflare, etc. Pass ``None``
        for plain AWS S3.
    region_name:
        AWS region. Inherited from the default boto3 chain if unset.
    client:
        Pre-built ``boto3.client("s3", ...)``. When supplied, the
        constructor skips creating its own client — handy for tests
        sharing one moto mock and for advanced callers wiring up
        custom retries, proxies, signature versions, etc.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        client: Any = None,
    ) -> None:
        if not _HAS_BOTO3:
            raise ImportError(
                'boto3 not installed; install with: pip install "soma[s3]"'
            )
        self._bucket: str = bucket
        # Normalise: strip leading + trailing slashes and backslashes so
        # ``prefix="/bundle/"`` and ``prefix="bundle"`` are equivalent
        # (callers read ``s3://bucket/prefix/`` and sometimes copy the
        # URL verbatim).
        self._prefix: str = prefix.replace("\\", "/").strip("/")
        if client is not None:
            self._client = client
        else:
            # boto3.client signature accepts None for optional kwargs.
            self._client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                region_name=region_name,
            )
        # Lazy staging state (see ``local_root``). Initialised to None
        # so the temp dir is only created on first access — callers that
        # only do direct put_bytes / get_bytes (the common Save/Load
        # path) never pay for it.
        self._local_root: Path | None = None
        self._staged_keys: set[str] = set()
        self._closed: bool = False
        # Register the sync-on-close hook so a crash before the caller
        # calls ``close()`` still pushes staged changes back to S3.
        atexit.register(self._atexit_close)

    # ------------------------------------------------------------------
    # Key / prefix plumbing
    # ------------------------------------------------------------------
    def _full_key(self, key: str) -> str:
        """Turn a caller-supplied key into an absolute S3 key.

        Normalises backslashes (Windows callers sometimes leak them
        through) and strips any leading slash so the key never
        accidentally escapes the configured prefix.
        """
        normalised = key.replace("\\", "/").lstrip("/")
        if self._prefix:
            return f"{self._prefix}/{normalised}"
        return normalised

    def _strip_prefix(self, full_key: str) -> str:
        """Remove the configured prefix from an absolute S3 key."""
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
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=full)
        except ClientError as exc:
            if _is_not_found(exc):
                raise KeyError(key) from exc
            raise
        body = resp["Body"]
        try:
            data = body.read()
        finally:
            body.close()
        return bytes(data)

    def put_bytes(self, key: str, data: bytes) -> None:
        full = self._full_key(key)
        self._client.put_object(Bucket=self._bucket, Key=full, Body=data)

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
        * Use the ``local_root`` staging path — mirrors the S3 prefix
          to a temp dir once, reads files with native ``open()``.
        * Call ``head_object`` and multi-range-GET themselves.

        The decision to slurp-into-BytesIO matches the local adapter's
        practical behaviour: ``open("rb")`` returns a seekable file
        handle and ``np.load`` / ``torch.load`` expect that contract.
        Without this the S3 path would silently fail the first time a
        user loads a bundle over ``s3://`` (torch's ``PyTorchFileReader``
        needs seek, and ``StreamingBody`` doesn't support it).
        """
        full = self._full_key(key)
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=full)
        except ClientError as exc:
            if _is_not_found(exc):
                raise KeyError(key) from exc
            raise
        body = resp["Body"]
        try:
            data = body.read()
        finally:
            with contextlib.suppress(Exception):
                body.close()
        return io.BytesIO(data)

    def put_stream(self, key: str, stream: BinaryIO) -> None:
        full = self._full_key(key)
        # ``upload_fileobj`` chunks through boto's TransferConfig (8 MB
        # default) so a hundred-MB payload streams up without RAM spikes.
        self._client.upload_fileobj(stream, self._bucket, full)

    # ------------------------------------------------------------------
    # Directory-style helpers
    # ------------------------------------------------------------------
    def list_prefix(self, prefix: str) -> Iterator[str]:
        """Yield keys under the store prefix that start with ``prefix``.

        Uses ``get_paginator("list_objects_v2")`` so bundles with more
        than 1000 objects don't silently truncate. ``.tmp`` siblings
        (see :class:`LocalFSObjectStore`) are filtered out for parity.
        """
        # S3 filters on the full key, so apply the store prefix once
        # and ask S3 to filter on the combined prefix.
        normalised = prefix.replace("\\", "/").lstrip("/")
        full_prefix = f"{self._prefix}/{normalised}" if self._prefix else normalised
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []) or []:
                full = obj["Key"]
                if full.endswith(".tmp"):
                    continue
                rel = self._strip_prefix(full)
                if rel.startswith(prefix):
                    yield rel

    def delete(self, key: str) -> None:
        """Remove ``key``. Idempotent — S3 ``delete_object`` on a missing
        key is already a no-op, so we just forward the call."""
        full = self._full_key(key)
        self._client.delete_object(Bucket=self._bucket, Key=full)

    def exists(self, key: str) -> bool:
        full = self._full_key(key)
        try:
            self._client.head_object(Bucket=self._bucket, Key=full)
        except ClientError as exc:
            if _is_not_found(exc):
                return False
            raise
        return True

    # ------------------------------------------------------------------
    # Local staging — lazy mirror of the S3 prefix as a real directory
    # ------------------------------------------------------------------
    @property
    def local_root(self) -> Path:
        """Lazy local mirror of the S3 prefix for WAL / lock use.

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
          file whose relative path corresponds to a key that was
          modified locally (we track these via :attr:`_staged_keys`).

        The invariant is "whatever is under ``local_root`` at close
        time is what ends up in S3". Calls that go directly through
        :meth:`put_bytes` / :meth:`get_bytes` bypass the staging dir
        (they talk to S3 directly) — the staging path is only walked
        by code that needs a filesystem handle.
        """
        if self._local_root is None:
            self._local_root = Path(
                tempfile.mkdtemp(prefix="soma-s3-stage-")
            )
            self._download_into_stage(self._local_root)
        return self._local_root

    def _download_into_stage(self, root: Path) -> None:
        """Mirror every object under the store prefix into ``root``.

        Forward-slash S3 keys map to ``os.sep``-separated directory
        hierarchies under ``root``. On Windows ``Path`` handles both
        directions transparently.
        """
        for rel_key in list(self.list_prefix("")):
            # ``list_prefix`` already returned POSIX-style keys relative
            # to our store prefix; join into the stage dir using Path
            # so platform separators land correctly.
            target = root / Path(rel_key)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = self.get_bytes(rel_key)
            except KeyError:
                # Raced with a concurrent delete — skip the vanished key.
                continue
            target.write_bytes(data)

    def close(self) -> None:
        """Push the local staging dir (if any) back to S3 and tear it down.

        Safe to call multiple times. When :attr:`local_root` was never
        accessed this is a no-op — no temp dir was created, nothing to
        sync. Otherwise every file under the stage dir is uploaded
        (overwriting any S3 object at the same key — we don't try to
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
                logger.warning("failed to clean up S3 staging dir %s", root)

    def _upload_from_stage(self, root: Path) -> None:
        """Upload every file under ``root`` back to S3 at the matching key.

        Uses ``put_object`` (bytes) rather than ``upload_fileobj``
        (threaded) because this path also runs from an ``atexit`` hook
        and boto's ``s3transfer`` thread pool refuses to schedule new
        work after interpreter shutdown. Staging files are typically
        small (WAL records, bundle.lock, index.json); the large-blob
        path goes through :meth:`put_bytes` / :meth:`put_stream`
        directly, not the stage dir.
        """
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                # Skip ``.tmp`` siblings: the local adapter's atomic
                # write lands a .tmp, then ``os.replace``s it — if we
                # ever stage via the local adapter path we must not
                # push the transient to S3.
                if fname.endswith(".tmp"):
                    continue
                abs_path = Path(dirpath) / fname
                rel = abs_path.relative_to(root).as_posix()
                # Read the full file and ``put_bytes`` rather than
                # ``put_stream``: atexit-safe (no thread pool), and
                # the files in the stage dir are the WAL sidecar and
                # friends — KB-MB at most.
                self.put_bytes(rel, abs_path.read_bytes())

    def _atexit_close(self) -> None:
        """atexit hook so a caller who forgot to call close() still flushes.

        Swallows every exception silently. By the time atexit runs the
        logger / stderr may already be closed, and we don't want our
        best-effort sync to foul the interpreter-shutdown path. In
        tests the moto mock is torn down before atexit fires and the
        client rejects with InvalidAccessKeyId — harmless.
        """
        with contextlib.suppress(Exception):  # pragma: no cover - best-effort
            self.close()


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------
def _is_not_found(exc: Exception) -> bool:
    """Recognise the "object does not exist" variants ``botocore`` emits.

    AWS S3 returns ``NoSuchKey`` for ``get_object`` and ``404`` for
    ``head_object``; MinIO and other S3-compat services sometimes
    surface just ``NotFound``. Match all three so callers get a clean
    ``KeyError`` regardless of which backend answered.
    """
    err = getattr(exc, "response", {}).get("Error", {}) if isinstance(exc, ClientError) else {}
    code = err.get("Code") if isinstance(err, dict) else None
    if code in ("NoSuchKey", "NotFound", "404"):
        return True
    # ``head_object`` 404s surface as ``Error.Code == '404'`` on newer
    # botocore; older releases stash the status under ResponseMetadata.
    meta = getattr(exc, "response", {}) if isinstance(exc, ClientError) else {}
    if not isinstance(meta, dict):
        return False
    status = meta.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status == 404


