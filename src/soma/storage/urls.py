"""URL-to-ObjectStore dispatch.

``parse_store_url`` is the one conversion point from a user-supplied
string (or ``Path``) to a concrete :class:`ObjectStore`. Callers that
already have an ``ObjectStore`` instance don't need this — they hand
it straight to :meth:`MemoryLayer.save` / :meth:`MemoryLayer.load`.

Dispatch table:
  * ``file://...`` — :class:`LocalFSObjectStore` (this phase).
  * ``s3://...``  — Phase 31 adds the adapter.
  * ``gs://...``  — Phase 32 adds the adapter.
  * No scheme (plain path) — auto-prefixed to ``file://`` and routed
    through the local adapter. Makes ``mem.save("/tmp/bundle")`` and
    ``mem.save("file:///tmp/bundle")`` interchangeable.

Windows path handling is load-bearing and subtle — see
:func:`_file_url_to_path` for the full tour of the four forms.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, unquote, urlparse

from soma.storage.base import ObjectStore
from soma.storage.local import LocalFSObjectStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


def parse_store_url(url: str | os.PathLike[str]) -> ObjectStore:
    """Resolve ``url`` to an :class:`ObjectStore`.

    Accepts:

    * A ``str`` URL (``file:///path/to/bundle``, or a plain
      ``/path/to/bundle`` / ``C:/path/to/bundle`` which is auto-prefixed
      to ``file://``; ``s3://bucket/prefix`` for S3-compatible stores).
    * A ``Path`` / ``PathLike``, treated as a local filesystem path.

    For ``s3://`` URLs, query-string parameters override the defaults
    threaded to :class:`S3ObjectStore`:

    * ``?region=eu-west-1`` — sets ``region_name``.
    * ``?endpoint=http://minio:9000`` — sets ``endpoint_url``, so the
      same adapter serves MinIO, LocalStack, Cloudflare R2,
      DigitalOcean Spaces.

    Raises ``ValueError`` with the literal message ``"unsupported
    store scheme: <scheme>"`` for any URL whose scheme is not yet
    wired into the dispatch table. Phase 32 flips ``gs://`` into a
    concrete adapter construction.
    """
    # Pathlib.Path / PathLike → treat as a local FS path unconditionally.
    if isinstance(url, os.PathLike):
        return LocalFSObjectStore(Path(url))

    if not isinstance(url, str):  # pragma: no cover — defensive only.
        raise TypeError(
            f"parse_store_url expects str or PathLike, got {type(url).__name__}"
        )
    if not url:
        raise ValueError("parse_store_url requires a non-empty URL")

    # Detect scheme the same way urlparse does. We roll our own scheme
    # sniff first because ``urlparse`` on Windows paths like
    # ``C:/Users/foo`` sees ``C`` as the scheme, which is not what we want.
    scheme = _sniff_scheme(url)

    if scheme is None:
        # Plain path, no scheme — auto-prefix to file://.
        return LocalFSObjectStore(Path(url))

    if scheme == "file":
        path = _file_url_to_path(url)
        return LocalFSObjectStore(path)

    if scheme == "s3":
        return _s3_url_to_store(url)

    raise ValueError(f"unsupported store scheme: {scheme}")


def _s3_url_to_store(url: str) -> ObjectStore:
    """Turn ``s3://bucket[/prefix][?region=...&endpoint=...]`` into a store.

    Split on the first ``/`` after the authority: everything before is
    the bucket, everything after is the prefix (trailing slash stripped).
    Query-string kwargs map to :class:`S3ObjectStore` constructor params.
    """
    # Import here so the core storage package doesn't hard-depend on
    # boto3 — users who never touch S3 don't need the extra installed.
    from soma.storage.s3 import S3ObjectStore

    parsed = urlparse(url)
    bucket = parsed.netloc
    if not bucket:
        raise ValueError(f"s3:// URL missing bucket: {url!r}")
    # ``parsed.path`` is ``"/prefix/..."`` or empty. Trim the leading
    # slash so the store's prefix stays the POSIX form we document.
    prefix = parsed.path.lstrip("/").rstrip("/")
    # Query-string kwargs. ``parse_qs`` returns ``dict[str, list[str]]``
    # — we take the first value of each.
    qs = parse_qs(parsed.query)
    region = qs.get("region", [None])[0]
    endpoint = qs.get("endpoint", [None])[0]
    return S3ObjectStore(
        bucket=bucket,
        prefix=prefix,
        endpoint_url=endpoint,
        region_name=region,
    )


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------
# Schemes we recognise — anything else is either auto-prefixed (no
# scheme) or raises. Single-letter "schemes" (``C:``) on Windows paths
# fall through ``_sniff_scheme`` as ``None`` so they get treated as
# local paths instead of raising.
_KNOWN_SCHEMES: frozenset[str] = frozenset({"file", "s3", "gs", "gcs", "http", "https"})


def _sniff_scheme(url: str) -> str | None:
    """Return the URL scheme, or ``None`` for plain paths.

    Handles the Windows-path edge case: ``urlparse("C:/Users/foo")``
    returns ``scheme='c'`` which is not a real URL scheme. We only
    honour a prefix as a scheme when it's followed by ``://`` (i.e.
    the authority form) or when it matches one of the known schemes.
    """
    # Case-insensitive match, but the scheme itself is lowercase.
    if "://" in url:
        head, _, _ = url.partition("://")
        return head.lower()
    if ":" in url:
        head, _, _ = url.partition(":")
        head_l = head.lower()
        # Single-letter prefixes are Windows drive letters; not schemes.
        if len(head_l) == 1:
            return None
        if head_l in _KNOWN_SCHEMES:
            return head_l
    return None


def _file_url_to_path(url: str) -> Path:
    """Convert a ``file://`` URL to a local :class:`Path`.

    Four forms we must handle:

    1. ``file:///abs/posix/path``  — RFC-canonical POSIX URL.
       ``urlparse`` → ``netloc=''``, ``path='/abs/posix/path'``.
    2. ``file:///C:/Users/foo``     — RFC-canonical Windows URL.
       ``urlparse`` → ``netloc=''``, ``path='/C:/Users/foo'``. The
       leading slash is cosmetic; the real path starts at ``C:``.
    3. ``file://C:/Users/foo``      — Windows-odd two-slash form.
       ``urlparse`` → ``netloc='C:'``, ``path='/Users/foo'``. We have
       to glue the drive letter back on.
    4. ``file://host/share/x``      — UNC share. Rare; we pass it
       through unchanged so ``Path`` sees ``\\\\host\\share\\x``.
    """
    parsed = urlparse(url)
    assert parsed.scheme == "file"

    netloc = parsed.netloc
    raw_path = unquote(parsed.path)

    # Case 3 — Windows drive letter landed in netloc.
    if netloc and len(netloc) == 2 and netloc[1] == ":":
        # Strip the leading slash on raw_path (urlparse always adds one
        # when netloc is present and the path is absolute). Glue back
        # to ``C:/Users/foo``.
        local = netloc + raw_path
        return Path(local)

    # Case 2 — ``/C:/...`` on any OS. The leading slash is cosmetic.
    if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
        return Path(raw_path[1:])

    # Case 4 — UNC-style file://host/share/file. Reconstruct as
    # ``\\host\share\file`` on Windows; leave as ``/share/file`` on POSIX
    # (the netloc would be a hostname that POSIX Path can't meaningfully
    # use — this is a Windows feature).
    if netloc and os.name == "nt":
        return Path(f"\\\\{netloc}{raw_path.replace('/', os.sep)}")

    # Case 1 — plain POSIX absolute path. On POSIX this is fine; on
    # Windows a POSIX absolute path with no drive letter is unusual
    # but ``Path`` will accept it (resolves against the current drive).
    return Path(raw_path)
