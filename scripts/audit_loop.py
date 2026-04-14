"""Phase 1 validation oracle for the autonomous improvement loop.

Runs the seven §6.2 checks and exits 1 if any fails. Used during the
48-hour validation run and in Phase transition runbooks.

Usage::

    python scripts/audit_loop.py [--since "48 hours ago"]
                                 [--gitea-base URL]
                                 [--gitea-token TOKEN]
                                 [--no-wiki-check]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---- Carveout pattern (G73) ------------------------------------------------

CARVEOUT_PATTERN = re.compile(
    r"^(src/soma/(core|memory|growth)/|"
    r"src/soma/system\.py$|"
    r"data/.*\.(txt|jsonl)$|"
    r"scripts/|"
    r"tests/|"
    r"pyproject\.toml$|"
    r"CLAUDE\.md$|"
    r"docs/plans/)"
)


# ---- git log parsing -------------------------------------------------------


_COMMIT_LINE = re.compile(r"^commit\s+([0-9a-f]+)")
_CHANGE_LOG_TRAILER = re.compile(r"^\s*Change-log-id:\s*(\S+)", re.IGNORECASE)


def parse_commit_log_entries(raw: str) -> list[dict[str, Any]]:
    """Parse ``git log`` raw output into ``[{sha, is_auto, change_log_id}, ...]``."""
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    subject_seen = False

    for line in raw.splitlines():
        m = _COMMIT_LINE.match(line)
        if m:
            if current is not None:
                entries.append(current)
            current = {
                "sha": m.group(1),
                "is_auto": False,
                "change_log_id": None,
            }
            subject_seen = False
            continue
        if current is None:
            continue
        if line.startswith("    ") or line.startswith("\t"):
            body = line.strip()
            if not subject_seen and body:
                current["is_auto"] = body.lower().startswith("auto:")
                subject_seen = True
            trailer = _CHANGE_LOG_TRAILER.match(body)
            if trailer:
                current["change_log_id"] = trailer.group(1)
    if current is not None:
        entries.append(current)
    return entries


def get_commit_log(since: str) -> str:
    proc = subprocess.run(
        ["git", "log", f"--since={since}", "--all"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


def get_commit_changed_paths(sha: str) -> set[str]:
    proc = subprocess.run(
        ["git", "show", "--name-only", "--pretty=format:", sha],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return set()
    return {ln.strip() for ln in proc.stdout.splitlines() if ln.strip()}


# ---- Individual checks -----------------------------------------------------


def check_auto_commits_have_change_log_id(
    commits: Sequence[dict[str, Any]],
    *,
    change_log_entries: Sequence[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Every ``^auto:`` commit must have a ``Change-log-id:`` trailer OR a
    change_log.jsonl entry referencing its SHA.
    """
    change_log_shas = {str(e.get("commit_sha")) for e in change_log_entries if e.get("commit_sha")}
    change_log_ids = {
        str(e.get("change_log_id")) for e in change_log_entries if e.get("change_log_id")
    }
    bad: list[str] = []
    for commit in commits:
        if not commit.get("is_auto"):
            continue
        sha = str(commit.get("sha", ""))
        trailer = commit.get("change_log_id")
        if trailer and trailer in change_log_ids:
            continue
        if trailer and not change_log_entries:
            # Trailer present, no log file available — trust the trailer.
            continue
        if sha and sha in change_log_shas:
            continue
        bad.append(sha)
    return (len(bad) == 0, bad)


def check_no_carveout_touch(
    touches: Iterable[tuple[str, bool, set[str]]],
) -> tuple[bool, list[tuple[str, list[str]]]]:
    """Flag any ``^auto:`` commit that touches a carveout path.

    Input: iterable of ``(sha, is_auto, paths)`` tuples.
    """
    bad: list[tuple[str, list[str]]] = []
    for sha, is_auto, paths in touches:
        if not is_auto:
            continue
        offenders = [p for p in paths if CARVEOUT_PATTERN.search(p)]
        if offenders:
            bad.append((sha, offenders))
    return (len(bad) == 0, bad)


def check_device_cuda_consistent(
    heartbeats: Sequence[dict[str, Any]],
) -> tuple[bool, str | None]:
    """No ``device == cpu`` entry in the heartbeat history."""
    for hb in heartbeats:
        device = str(hb.get("device", ""))
        if device.startswith("cpu"):
            return False, f"heartbeat shows cpu at ts={hb.get('ts')}"
    return True, None


def check_disk_footprint(directory: Path, *, max_gb: float) -> tuple[bool, float]:
    """Sum bytes under ``directory`` (recursively) and compare to ``max_gb``.

    Missing directory counts as 0 bytes. Individual file-stat failures are
    silently skipped — this is an audit, not a filesystem integrity check.
    """
    if not directory.exists():
        return True, 0.0
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    size_gb = total / (1024 * 1024 * 1024)
    return (size_gb <= max_gb, size_gb)


def check_consecutive_failures_zero(path: Path) -> tuple[bool, str | None]:
    if not path.exists():
        return True, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"could not read {path}: {exc}"
    count = int(data.get("count", 0))
    if count != 0:
        return False, f"count={count}"
    return True, None


def check_gate_passed_on_pristine(
    *,
    gate_failure: Path,
    change_log_entries: Sequence[dict[str, Any]],
) -> tuple[bool, str | None]:
    """Return True if there is no recent gate failure, or if the loop has
    since recorded a confirmed change (which means the gate must have
    accepted that change)."""
    if not gate_failure.exists():
        return True, None

    for entry in change_log_entries:
        if entry.get("status") == "confirmed":
            confirmed_at = entry.get("confirmed_at")
            failure_ts_raw = 0.0
            try:
                failure_ts_raw = float(
                    json.loads(gate_failure.read_text(encoding="utf-8")).get("ts", 0)
                )
            except (json.JSONDecodeError, OSError, TypeError):
                failure_ts_raw = 0.0
            try:
                if confirmed_at and float(confirmed_at) >= failure_ts_raw:
                    return True, None
            except (TypeError, ValueError):
                continue
    return False, f"gate failure recorded at {gate_failure}"


def check_wiki_pushes(
    *,
    base_url: str | None,
    token: str | None,
    repo: str,
    since_ts: float,
    min_pushes: int = 5,
) -> tuple[bool, str | None]:
    """Query Gitea for SOMA-session-* conversation commits since ``since_ts``.

    If ``base_url`` or ``token`` is missing we treat the check as skipped
    (True, "skipped") — the operator explicitly opts out by omitting either.
    """
    if not base_url or not token:
        return True, "skipped (no Gitea credentials)"
    import urllib.error
    import urllib.parse
    import urllib.request

    since_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(since_ts))
    params = urllib.parse.urlencode({"since": since_iso, "path": "conversations", "limit": 50})
    url = f"{base_url.rstrip('/')}/api/v1/repos/{repo}/commits?{params}"
    req = urllib.request.Request(url, headers={"Authorization": f"token {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (internal API)
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        return False, f"Gitea API unreachable: {exc}"
    except json.JSONDecodeError as exc:
        return False, f"Gitea returned non-JSON: {exc}"
    pushes = 0
    for commit in payload if isinstance(payload, list) else []:
        files = commit.get("files") or []
        for f in files:
            name = str(f.get("filename", ""))
            if "SOMA-session-" in name:
                pushes += 1
                break
    if pushes < min_pushes:
        return False, f"only {pushes} wiki pushes (need {min_pushes})"
    return True, None


# ---- Main ------------------------------------------------------------------


def _load_change_log_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _load_heartbeats(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [data]


def _print_result(name: str, ok: bool, detail: str) -> None:
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SOMA audit oracle.")
    parser.add_argument("--since", default="48 hours ago")
    parser.add_argument(
        "--gitea-base",
        default=os.environ.get("SOMA_GITEA_BASE"),
        help="Gitea base URL for wiki-push check (e.g., http://localhost:3000)",
    )
    parser.add_argument(
        "--gitea-token",
        default=os.environ.get("SOMA_GITEA_TOKEN"),
        help="Gitea API token for wiki-push check.",
    )
    parser.add_argument(
        "--gitea-repo",
        default=os.environ.get("SOMA_GITEA_REPO", "dant123/knowledge-wiki"),
    )
    parser.add_argument("--no-wiki-check", action="store_true")
    parser.add_argument("--max-gb", type=float, default=10.0)
    args = parser.parse_args(argv)

    repo_root = Path(".")
    state_dir = repo_root / ".soma-loop/state"
    change_log = state_dir / "change_log.jsonl"
    heartbeat = state_dir / "train_heartbeat.json"
    consecutive_fail = state_dir / "consecutive_failures.json"
    gate_failure = state_dir / "gate_failure.json"
    soma_loop_dir = repo_root / ".soma-loop"
    checkpoints_dir = repo_root / "checkpoints"

    print(f"audit_loop: checking {args.since}")

    change_log_entries = _load_change_log_entries(change_log)
    heartbeats = _load_heartbeats(heartbeat)

    # 1. auto commits have change_log trailer
    commits = parse_commit_log_entries(get_commit_log(args.since))
    ok_trailer, bad_trailer = check_auto_commits_have_change_log_id(
        commits, change_log_entries=change_log_entries
    )
    _print_result(
        "auto-commits-have-trailer",
        ok_trailer,
        "all match" if ok_trailer else f"{len(bad_trailer)} unmatched: {bad_trailer[:5]}",
    )

    # 2. no carveout touches
    touches: list[tuple[str, bool, set[str]]] = []
    for commit in commits:
        touches.append(
            (commit["sha"], bool(commit["is_auto"]), get_commit_changed_paths(commit["sha"]))
        )
    ok_carve, bad_carve = check_no_carveout_touch(touches)
    _print_result(
        "no-carveout-touches",
        ok_carve,
        "clean" if ok_carve else f"violations: {bad_carve[:3]}",
    )

    # 3. device consistency
    ok_device, device_reason = check_device_cuda_consistent(heartbeats)
    _print_result(
        "device-cuda-consistent",
        ok_device,
        "cuda" if ok_device else (device_reason or "unknown"),
    )

    # 4. disk footprint
    total_gb = 0.0
    ok_disk_loop, loop_gb = check_disk_footprint(soma_loop_dir, max_gb=args.max_gb)
    ok_disk_ckpt, ckpt_gb = check_disk_footprint(checkpoints_dir, max_gb=args.max_gb)
    total_gb = loop_gb + ckpt_gb
    ok_disk = ok_disk_loop and ok_disk_ckpt and (total_gb <= args.max_gb)
    _print_result(
        "disk-footprint",
        ok_disk,
        f".soma-loop={loop_gb:.2f}GB ckpt={ckpt_gb:.2f}GB total={total_gb:.2f}GB",
    )

    # 5. consecutive_failures == 0
    ok_cf, cf_reason = check_consecutive_failures_zero(consecutive_fail)
    _print_result("consecutive-failures-zero", ok_cf, cf_reason or "count=0")

    # 6. gate passed on pristine
    ok_gate, gate_reason = check_gate_passed_on_pristine(
        gate_failure=gate_failure, change_log_entries=change_log_entries
    )
    _print_result(
        "gate-passed-pristine", ok_gate, "no stale failure" if ok_gate else (gate_reason or "")
    )

    # 7. wiki pushes
    ok_wiki: bool
    wiki_reason: str | None
    if args.no_wiki_check:
        ok_wiki, wiki_reason = True, "skipped (--no-wiki-check)"
    else:
        # Compute since_ts from args.since if possible — else use 48h back.
        since_ts = time.time() - 48 * 3600
        ok_wiki, wiki_reason = check_wiki_pushes(
            base_url=args.gitea_base,
            token=args.gitea_token,
            repo=args.gitea_repo,
            since_ts=since_ts,
        )
    _print_result("wiki-pushes", ok_wiki, wiki_reason or "sufficient")

    all_ok = all([ok_trailer, ok_carve, ok_device, ok_disk, ok_cf, ok_gate, ok_wiki])
    if not all_ok:
        print("audit_loop: FAIL", file=sys.stderr)
        return 1
    print("audit_loop: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
