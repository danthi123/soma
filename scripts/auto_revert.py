"""Autonomous-loop watchdog — confirms or reverts the last in-progress change.

Invoked every ~2 minutes by an external scheduler. Produces no output when
there is nothing to do. On confirmation: flips the change_log entry to
``confirmed`` and copies ``current.pt → last_good.pt``. On revert: runs
``git revert --no-edit``, restores the last-good checkpoint, signals
training to shut down so the tick can restart.

See docs/plans/2026-04-12-autonomous-loop-design.md §5.2 ``auto_revert.py``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lock import FileLock

WINDOW_MIN = 15
MIN_STEP_ADVANCE = 500
TOLERANCE_PCT = 0.20
CONFIRMATION_BUCKETS_REQUIRED = 15
REVERT_CONSECUTIVE_BAD_BUCKETS = 3
STALE_TICK_SECONDS = 3600


# ---- Pure helpers ----------------------------------------------------------


def is_stale_tick(last_tick: dict[str, Any]) -> bool:
    ts_raw = last_tick.get("ts")
    if ts_raw is None:
        return True
    try:
        ts = float(ts_raw)
    except (TypeError, ValueError):
        return True
    return (time.time() - ts) > STALE_TICK_SECONDS


def read_recent_metrics(path: Path, *, since_ts: float) -> list[dict[str, Any]]:
    """Return records with ``ts >= since_ts`` from the metrics JSONL file."""
    if not path.exists():
        return []
    recent: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                ts = float(rec.get("ts", 0))
            except (TypeError, ValueError):
                continue
            if ts >= since_ts:
                recent.append(rec)
    return recent


def bucket_by_minute(records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group records into per-minute buckets (oldest first).

    A single bucket holds every record whose ``floor(ts / 60)`` matches.
    """
    if not records:
        return []
    buckets: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        try:
            minute_key = int(float(rec.get("ts", 0)) // 60)
        except (TypeError, ValueError):
            continue
        buckets.setdefault(minute_key, []).append(rec)
    return [buckets[k] for k in sorted(buckets.keys())]


def service_crashed_after_apply(
    perm_fail_path: Path, *, ts_applied: float
) -> bool:
    """Return True if ``perm_fail_path`` exists and post-dates ``ts_applied``.

    train_service writes ``train_permanent_failure.json`` after 3 uncaught
    crashes in 5 minutes. If it post-dates the current in-flight change's
    apply-time, the change killed the service and we should revert
    immediately (no metrics will ever arrive to regression-judge against).
    """
    if not perm_fail_path.exists():
        return False
    try:
        perm_fail = json.loads(perm_fail_path.read_text(encoding="utf-8"))
        perm_fail_ts = float(perm_fail.get("ts", 0) or 0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return perm_fail_ts > ts_applied


def check_confirmation(
    pre: dict[str, Any],
    recent: Sequence[dict[str, Any]],
    *,
    window_min: int,
    min_step_advance: int,
) -> tuple[str, str | None, dict[str, bool]]:
    """Evaluate confirm/revert/pending verdict over ``recent`` metrics.

    Verdicts:
      * ``"confirmed"``: every criterion held for ``window_min`` minute-buckets
      * ``"revert"``:    some criterion failed for
                         ``REVERT_CONSECUTIVE_BAD_BUCKETS`` consecutive buckets
      * ``"pending"``:   not enough data yet

    Criteria (per-bucket averages):
      * ``loss_within_20pct``
      * ``curiosity_within_20pct``
      * ``no_nan_inf``
      * ``nodes_not_collapsed`` (>= 50% of pre-change node count)
      * ``step_advancing`` (handled outside the per-bucket loop)
    """
    buckets = bucket_by_minute(recent)
    if len(buckets) < 2:
        return "pending", "insufficient buckets", {}

    last_step = max(int(r.get("step", 0)) for r in recent)
    first_step = min(int(r.get("step", 0)) for r in recent)
    if last_step - first_step < min_step_advance:
        return (
            "pending",
            f"step advance {last_step - first_step} < {min_step_advance}",
            {},
        )

    pre_loss_raw = pre.get("loss")
    pre_loss = float(pre_loss_raw) if pre_loss_raw is not None else None
    pre_cur = float(pre.get("curiosity", 0.5))
    pre_nodes = int(pre.get("num_nodes", 0))

    def bucket_pass(bucket: list[dict[str, Any]]) -> tuple[bool, dict[str, bool]]:
        losses: list[float] = []
        for r in bucket:
            loss_raw = r.get("loss")
            if loss_raw is None:
                continue
            try:
                losses.append(float(loss_raw))
            except (TypeError, ValueError):
                continue
        curs = [float(r.get("curiosity", 0.5)) for r in bucket]
        nodes = [int(r.get("num_nodes", 0)) for r in bucket]

        loss_within: bool
        if pre_loss is None or not losses:
            loss_within = True
        else:
            avg_loss = sum(losses) / len(losses)
            loss_within = (
                not math.isnan(avg_loss)
                and abs(avg_loss - pre_loss) <= TOLERANCE_PCT * abs(pre_loss or 1.0)
            )

        avg_cur = (sum(curs) / len(curs)) if curs else pre_cur
        curiosity_within = abs(avg_cur - pre_cur) <= TOLERANCE_PCT * abs(pre_cur or 1.0)

        no_nan_inf = all(not math.isnan(x) and not math.isinf(x) for x in losses)

        min_nodes = min(nodes) if nodes else pre_nodes
        nodes_not_collapsed = (
            True if pre_nodes == 0 else min_nodes >= int(0.5 * pre_nodes)
        )

        crits = {
            "loss_within_20pct": loss_within,
            "curiosity_within_20pct": curiosity_within,
            "no_nan_inf": no_nan_inf,
            "nodes_not_collapsed": nodes_not_collapsed,
            "step_advancing": True,
        }
        return all(crits.values()), crits

    bad_streak = 0
    good_streak = 0
    latest_criteria: dict[str, bool] = {}
    for bucket in buckets:
        passed, crits = bucket_pass(bucket)
        latest_criteria = crits
        if passed:
            bad_streak = 0
            good_streak += 1
        else:
            bad_streak += 1
            good_streak = 0
            if bad_streak >= REVERT_CONSECUTIVE_BAD_BUCKETS:
                return (
                    "revert",
                    f"criteria failed for {bad_streak} consecutive minute-buckets",
                    crits,
                )

    if good_streak >= CONFIRMATION_BUCKETS_REQUIRED and len(buckets) >= window_min:
        return "confirmed", "all criteria held over confirmation window", latest_criteria

    return "pending", "within confirmation window", latest_criteria


# ---- Change log I/O -------------------------------------------------------


def append_change_log_entry(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def update_change_log_entry(
    path: Path, *, change_log_id: str, updates: dict[str, Any]
) -> None:
    """Rewrite the single matching entry in-place.

    The change log is bounded by tick rate limits, so rewriting the full file
    is cheap.
    """
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
        if rec.get("change_log_id") == change_log_id:
            rec.update(updates)
        new_lines.append(json.dumps(rec))
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _summarize_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    losses: list[float] = []
    for r in records:
        val = r.get("loss")
        if val is None:
            continue
        try:
            losses.append(float(val))
        except (TypeError, ValueError):
            continue
    last_loss: float | None
    if not records:
        last_loss = None
    else:
        raw = records[-1].get("loss")
        try:
            last_loss = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            last_loss = None
    return {
        "count": len(records),
        "last_loss": last_loss,
        "mean_loss": (sum(losses) / len(losses)) if losses else None,
    }


# ---- Main -----------------------------------------------------------------


def _parse_ts(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return datetime.fromisoformat(value).timestamp()
        with contextlib.suppress(ValueError):
            return float(value)
    return 0.0


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _load_consecutive_failures(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return {"count": 0, "last_reset_ts": None, "last_failure_ts": None}


def _save_consecutive_failures(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _find_last_in_progress(lines: list[str]) -> dict[str, Any] | None:
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "in_progress" and rec.get("commit_sha"):
            return dict(rec)
    return None


def main() -> int:
    repo = Path(".")
    state_dir = repo / ".soma-loop/state"
    metrics_path = repo / ".soma-loop/metrics/metrics.current.jsonl"
    last_tick_path = state_dir / "last_tick.json"
    change_log = state_dir / "change_log.jsonl"
    consecutive_fail = state_dir / "consecutive_failures.json"
    git_lock = state_dir / "git.lock"
    revert_failure = state_dir / "revert_failure.json"
    stop_flag = repo / ".soma-loop/STOP"

    if not last_tick_path.exists():
        return 0
    try:
        last_tick = json.loads(last_tick_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if is_stale_tick(last_tick):
        return 0

    if not change_log.exists():
        return 0
    lines = [
        ln for ln in change_log.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    target = _find_last_in_progress(lines)
    if target is None:
        return 0

    since_ts = _parse_ts(target.get("ts_applied") or target.get("ts_proposed") or 0)

    # Service-crash short-circuit: if train_service hit its permanent
    # failure after the change was applied, there will never be post-change
    # metrics to regression-judge against. Revert immediately.
    perm_fail_path = state_dir / "train_permanent_failure.json"
    if service_crashed_after_apply(perm_fail_path, ts_applied=since_ts):
        verdict, reason = "revert", "train_service permanent_failure after change applied"
        criteria: list[str] = ["service_dead"]
        recent: list[dict[str, Any]] = []
    else:
        recent = read_recent_metrics(metrics_path, since_ts=since_ts)
        if not recent:
            return 0
        first_step = min(int(r.get("step", 0)) for r in recent)
        last_step = max(int(r.get("step", 0)) for r in recent)
        if last_step == first_step:
            # Training paused — defer judgment per design §5.2 G30.
            return 0
        pre = dict(target.get("pre_change_metrics", {}) or {})
        verdict, reason, criteria = check_confirmation(
            pre, recent, window_min=WINDOW_MIN, min_step_advance=MIN_STEP_ADVANCE
        )
        if verdict == "pending":
            return 0

    change_log_id = str(target["change_log_id"])
    commit_sha = str(target["commit_sha"])

    lock = FileLock(git_lock, timeout=60.0, purpose=f"auto_revert:{verdict}")
    if not lock.acquire():
        print(
            f"auto_revert: could not acquire git.lock ({verdict})",
            file=sys.stderr,
        )
        return 0

    try:
        if verdict == "confirmed":
            update_change_log_entry(
                change_log,
                change_log_id=change_log_id,
                updates={
                    "status": "confirmed",
                    "confirmed_at": time.time(),
                    "post_change_metrics": _summarize_metrics(recent),
                },
            )
            cp_cur = Path("checkpoints/current.pt")
            cp_lg = Path("checkpoints/last_good.pt")
            if cp_cur.exists():
                _atomic_copy(cp_cur, cp_lg)
            # Promote the encoder sidecar too so a future revert can load
            # the matching-shape embeddings. Without this, falling back
            # to last_good.pt on a future revert would either find no
            # encoder sidecar or find the stale current.encoder.pt that
            # was saved against a later step.
            enc_cur = Path("checkpoints/current.encoder.pt")
            enc_lg = Path("checkpoints/last_good.encoder.pt")
            if enc_cur.exists():
                _atomic_copy(enc_cur, enc_lg)
            _save_consecutive_failures(
                consecutive_fail,
                {
                    "count": 0,
                    "last_reset_ts": time.time(),
                    "last_failure_ts": None,
                },
            )
            print(f"auto_revert: confirmed {commit_sha[:8]}")
        elif verdict == "revert":
            rev_result = subprocess.run(
                ["git", "revert", "--no-edit", commit_sha],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if rev_result.returncode != 0:
                revert_failure.parent.mkdir(parents=True, exist_ok=True)
                revert_failure.write_text(
                    json.dumps(
                        {
                            "change_log_id": change_log_id,
                            "commit_sha": commit_sha,
                            "error": rev_result.stderr[:4000],
                            "ts": time.time(),
                        }
                    ),
                    encoding="utf-8",
                )
                stop_flag.touch()
                print(
                    f"auto_revert: revert FAILED, STOP set: {rev_result.stderr}",
                    file=sys.stderr,
                )
                return 1
            subprocess.run(
                [
                    "git",
                    "commit",
                    "--amend",
                    "-m",
                    f"revert(auto): revert {commit_sha[:8]} ({reason})",
                ],
                check=False,
                timeout=30,
            )
            update_change_log_entry(
                change_log,
                change_log_id=change_log_id,
                updates={
                    "status": "reverted",
                    "reverted_at": time.time(),
                    "reverted_by": "watchdog",
                    "failure_signal": criteria,
                    "failure_reason": reason,
                    "post_change_metrics": _summarize_metrics(recent),
                },
            )
            cf = _load_consecutive_failures(consecutive_fail)
            cf["count"] = int(cf.get("count", 0)) + 1
            cf["last_failure_ts"] = time.time()
            _save_consecutive_failures(consecutive_fail, cf)

            cp_cur = Path("checkpoints/current.pt")
            cp_lg = Path("checkpoints/last_good.pt")
            if cp_lg.exists():
                _atomic_copy(cp_lg, cp_cur)
            # Restore matching encoder weights. Leaving current.encoder.pt
            # from the post-regression period would mean the just-reverted
            # SOMA graph is evaluated against embeddings it hasn't seen.
            enc_cur = Path("checkpoints/current.encoder.pt")
            enc_lg = Path("checkpoints/last_good.encoder.pt")
            if enc_lg.exists():
                _atomic_copy(enc_lg, enc_cur)
            (Path(".soma-loop/signals") / "shutdown").touch()
            print(f"auto_revert: reverted {commit_sha[:8]} ({reason})")
    finally:
        lock.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
