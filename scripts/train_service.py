"""Long-lived SOMA training daemon for the autonomous loop.

Responsibilities (across Tasks 5-10):
- Load config from configs/current.yaml (fallback: configs/default.yaml)
- Checkpoint-load fallback chain: current.pt -> last_good.pt -> fresh SOMA
- Write heartbeat to .soma-loop/state/train_heartbeat.json every 10s
- Poll .soma-loop/signals/ for pause/resume/shutdown/reload_config
- Append metrics every 10 steps
- Atomic checkpoint every config.checkpoint_interval steps
- Retention: last 20 step checkpoints + every-100th-checkpoint permanently
- Crash backoff: 3 exceptions within 5 min -> write train_permanent_failure.json + exit

Ctrl+C / SIGINT also triggers graceful shutdown.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lock import is_pid_alive


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON to path atomically via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


class PidFile:
    """Manages a PID file; atomic write, caller removes on exit."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
        os.replace(tmp_name, self.path)

    def remove(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()


def check_pid_collision(pid_path: Path) -> bool:
    """Return True if a live process already owns the pid file."""
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return False
    return is_pid_alive(pid)


# ---- Heartbeat --------------------------------------------------------------

HeartbeatStatus = Literal["warming_up", "running", "paused", "shutdown"]

HEARTBEAT_STATUS_WARMING_UP: HeartbeatStatus = "warming_up"
HEARTBEAT_STATUS_RUNNING: HeartbeatStatus = "running"
HEARTBEAT_STATUS_PAUSED: HeartbeatStatus = "paused"
HEARTBEAT_STATUS_SHUTDOWN: HeartbeatStatus = "shutdown"


class Heartbeat:
    """Writes a JSON heartbeat file atomically with current step + status."""

    def __init__(self, path: Path, *, device: str) -> None:
        self.path = path
        self.device = device

    def update(self, *, step: int, status: HeartbeatStatus) -> None:
        atomic_write_json(
            self.path,
            {
                "ts": time.time(),
                "step": step,
                "status": status,
                "device": self.device,
                "pid": os.getpid(),
            },
        )


# ---- Signals ----------------------------------------------------------------


class SignalPoller:
    """Filesystem sentinel poller. Signals consumed by deleting the file."""

    KNOWN_SIGNALS = frozenset({"shutdown", "pause", "resume", "reload_config"})

    def __init__(self, signals_dir: Path) -> None:
        self.signals_dir = signals_dir

    def read_and_consume(self) -> list[str]:
        if not self.signals_dir.exists():
            return []
        seen: list[str] = []
        for name in self.KNOWN_SIGNALS:
            path = self.signals_dir / name
            if path.exists():
                seen.append(name)
                with contextlib.suppress(OSError):
                    path.unlink()
        return seen


# ---- Checkpoint fallback chain ---------------------------------------------


@dataclass
class LoadResult:
    source: Literal["current", "last_good", "fresh"]
    payload: object | None


def _default_torch_loader(path: Path) -> object:
    import torch

    return torch.load(path, map_location="cpu")


def load_checkpoint_chain(
    *,
    current: Path,
    last_good: Path,
    fresh_flag: Path,
    loader: Callable[[Path], object] | None = None,
) -> LoadResult:
    """Try ``current`` → ``last_good`` → fresh init.

    ``loader`` exists for testing; in production it calls into ``torch.load``.
    Writes ``fresh_flag`` when falling through to fresh init.
    """
    load_fn = loader if loader is not None else _default_torch_loader

    for path, source in [(current, "current"), (last_good, "last_good")]:
        if path.exists():
            try:
                payload = load_fn(path)
                return LoadResult(source=source, payload=payload)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001 — intentionally broad: any load failure triggers fallback
                print(
                    f"train_service: {source} load failed ({exc}); trying fallback",
                    file=sys.stderr,
                )
                continue

    fresh_flag.parent.mkdir(parents=True, exist_ok=True)
    fresh_flag.touch()
    return LoadResult(source="fresh", payload=None)


# ---- Crash backoff ----------------------------------------------------------


class CrashBackoff:
    """Track crashes in a sliding time window; flag permanent failure beyond threshold."""

    def __init__(self, window_seconds: float = 300.0, max_crashes: int = 3) -> None:
        self.window_seconds = float(window_seconds)
        self.max_crashes = int(max_crashes)
        self.crashes: list[float] = []

    def record_crash(self, *, now: float | None = None) -> bool:
        """Record a crash at ``now`` (default: current time).

        Returns True if the number of crashes within the window is >= max_crashes.
        """
        ts = time.time() if now is None else float(now)
        self.crashes = [t for t in self.crashes if ts - t < self.window_seconds]
        self.crashes.append(ts)
        return len(self.crashes) >= self.max_crashes


# ---- Metrics append --------------------------------------------------------


def append_metric_record(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to the metrics JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---- Checkpoint retention --------------------------------------------------


_STEP_CKPT_PREFIX = "step_"
_STEP_CKPT_SUFFIX = ".pt"


def _parse_step(path: Path) -> int:
    """Return the step number embedded in ``step_NNNNNNNN.pt``, or -1."""
    if not path.name.startswith(_STEP_CKPT_PREFIX) or not path.name.endswith(
        _STEP_CKPT_SUFFIX
    ):
        return -1
    middle = path.stem[len(_STEP_CKPT_PREFIX) :]
    try:
        return int(middle)
    except ValueError:
        return -1


def prune_step_checkpoints(
    ckpt_dir: Path,
    *,
    keep_last: int = 20,
    permanent_every_steps: int = 500_000,
) -> list[Path]:
    """Delete step checkpoints beyond the last ``keep_last``, except steps
    divisible by ``permanent_every_steps`` which are always retained.

    ``current.pt`` and ``last_good.pt`` are never touched. Returns the list of
    deleted paths.
    """
    checkpoints: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob(f"{_STEP_CKPT_PREFIX}*{_STEP_CKPT_SUFFIX}"):
        step = _parse_step(p)
        if step >= 0:
            checkpoints.append((step, p))
    checkpoints.sort()
    if not checkpoints:
        return []

    keep_set: set[Path] = {p for _, p in checkpoints[-keep_last:]}
    if permanent_every_steps > 0:
        for step, p in checkpoints:
            if step % permanent_every_steps == 0:
                keep_set.add(p)

    deleted: list[Path] = []
    for _, p in checkpoints:
        if p not in keep_set:
            try:
                p.unlink()
                deleted.append(p)
            except OSError:
                continue
    return deleted


# ---- Main loop --------------------------------------------------------------


WARMUP_STEPS = 100
METRICS_APPEND_EVERY = 10
HEARTBEAT_INTERVAL_S = 10.0
CRASH_BACKOFF_SLEEP_S = 10.0
PAUSED_SLEEP_S = 0.5
KEEP_LAST_CKPT = 20


def _resolve_device(choice: str) -> Any:
    """Return a ``torch.device`` honoring ``auto`` (cuda if available else cpu)."""
    import torch

    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy ``src`` → ``dst`` atomically via ``dst.tmp`` + rename."""
    import shutil

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _save_checkpoint(
    soma: Any,
    ckpt_dir: Path,
    step: int,
    *,
    permanent_every_steps: int,
) -> Path:
    """Save SOMA to ``step_NNNNNNNN.pt`` atomically and update ``current.pt``
    + ``current.txt`` pointer. Prunes old step checkpoints.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step_path = ckpt_dir / f"step_{step:08d}.pt"
    step_tmp = step_path.with_suffix(step_path.suffix + ".tmp")
    soma.save_state(step_tmp)
    os.replace(step_tmp, step_path)

    _atomic_copy(step_path, ckpt_dir / "current.pt")

    pointer = ckpt_dir / "current.txt"
    pointer_tmp = pointer.with_suffix(pointer.suffix + ".tmp")
    pointer_tmp.write_text(step_path.name, encoding="utf-8")
    os.replace(pointer_tmp, pointer)

    prune_step_checkpoints(
        ckpt_dir,
        keep_last=KEEP_LAST_CKPT,
        permanent_every_steps=permanent_every_steps,
    )
    return step_path


def _build_metric_record(result: dict[str, Any], wm_occ: float, ep_count: int) -> dict[str, Any]:
    return {
        "ts": time.time(),
        "step": int(result.get("global_step", 0)),
        "loss": float(result["loss"]) if result.get("loss") is not None else None,
        "curiosity": float(result.get("curiosity", 0.0)),
        "lr_multiplier": float(result.get("lr_multiplier", 1.0)),
        "num_nodes": int(result.get("num_nodes", 0)),
        "num_edges": int(result.get("num_edges", 0)),
        "wm_occupancy": float(wm_occ),
        "episodic_entries": int(ep_count),
    }


def _iter_sample_pairs(feeder: Any, first_out: str) -> Any:
    """Yield (inputs, targets) dicts from the feeder, token-by-token."""
    while True:
        for sample in feeder:
            if sample.target.numel() == 0:
                continue
            target_len = sample.target.shape[0]
            lengths = {
                m: t.shape[0] for m, t in sample.inputs.items() if t.numel() > 0
            }
            if not lengths:
                continue
            for t in range(target_len):
                step_inputs: dict[str, Any] = {}
                for modality, seq in sample.inputs.items():
                    if modality not in lengths:
                        continue
                    idx = min(t, lengths[modality] - 1)
                    step_inputs[modality] = seq[idx].detach()
                if not step_inputs:
                    continue
                yield step_inputs, {first_out: sample.target[t].detach()}


def run_training_loop(
    *,
    soma: Any,
    feeder: Any,
    config: Any,
    heartbeat: Heartbeat,
    poller: SignalPoller,
    metrics_path: Path,
    ckpt_dir: Path,
    state_dir: Path,
    fresh_init: bool,
    max_steps: int | None = None,
) -> int:
    """Run the main training loop until shutdown, permanent failure, or step cap.

    Returns the process exit code (0 normal, 2 permanent failure).
    """
    first_out = config.output_modalities[0]
    pair_iter = _iter_sample_pairs(feeder, first_out)

    warmup_remaining = WARMUP_STEPS if fresh_init else 0
    paused = False
    crash_backoff = CrashBackoff()
    last_heartbeat_ts = 0.0
    last_metrics_step = soma.global_step
    permanent_every = 100 * int(config.checkpoint_interval)
    steps_taken = 0

    while True:
        # ---- Signal polling ----
        for sig in poller.read_and_consume():
            if sig == "shutdown":
                try:
                    _save_checkpoint(
                        soma, ckpt_dir, soma.global_step, permanent_every_steps=permanent_every
                    )
                except Exception as exc:  # noqa: BLE001 — shutdown save is best-effort
                    print(f"train_service: shutdown save failed: {exc}", file=sys.stderr)
                heartbeat.update(step=soma.global_step, status=HEARTBEAT_STATUS_SHUTDOWN)
                return 0
            if sig == "pause":
                paused = True
            elif sig == "resume":
                paused = False
            elif sig == "reload_config":
                # Phase 1: treat as shutdown so the scheduler restarts with new config.
                try:
                    _save_checkpoint(
                        soma, ckpt_dir, soma.global_step, permanent_every_steps=permanent_every
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"train_service: reload save failed: {exc}", file=sys.stderr)
                heartbeat.update(step=soma.global_step, status=HEARTBEAT_STATUS_SHUTDOWN)
                return 0

        # ---- Heartbeat ----
        now = time.time()
        if now - last_heartbeat_ts >= HEARTBEAT_INTERVAL_S:
            if paused:
                status: HeartbeatStatus = HEARTBEAT_STATUS_PAUSED
            elif warmup_remaining > 0:
                status = HEARTBEAT_STATUS_WARMING_UP
            else:
                status = HEARTBEAT_STATUS_RUNNING
            heartbeat.update(step=soma.global_step, status=status)
            last_heartbeat_ts = now

        if paused:
            time.sleep(PAUSED_SLEEP_S)
            continue

        # ---- Fetch next training pair ----
        try:
            inputs, targets = next(pair_iter)
        except StopIteration:
            time.sleep(0.1)
            continue

        # ---- Step with crash backoff ----
        try:
            result = soma.step(inputs=inputs, targets=targets)
        except Exception as exc:  # noqa: BLE001 — we catch everything and decide backoff vs exit
            crash_info = {
                "ts": time.time(),
                "step": soma.global_step,
                "error": str(exc),
                "type": type(exc).__name__,
            }
            atomic_write_json(state_dir / "train_crash.json", crash_info)
            print(f"train_service: step crashed: {exc}", file=sys.stderr)
            if crash_backoff.record_crash():
                atomic_write_json(
                    state_dir / "train_permanent_failure.json",
                    {
                        "ts": time.time(),
                        "step": soma.global_step,
                        "crashes_in_window": len(crash_backoff.crashes),
                        "window_seconds": crash_backoff.window_seconds,
                    },
                )
                return 2
            time.sleep(CRASH_BACKOFF_SLEEP_S)
            continue

        if warmup_remaining > 0:
            warmup_remaining -= 1

        steps_taken += 1

        # ---- Metrics append (cadence driven by soma.global_step) ----
        if soma.global_step - last_metrics_step >= METRICS_APPEND_EVERY:
            wm_occ = float(soma.working_memory.occupancy())
            ep_count = int(soma.episodic_memory.num_valid)
            record = _build_metric_record(result, wm_occ, ep_count)
            append_metric_record(metrics_path, record)
            last_metrics_step = soma.global_step

        # ---- Periodic checkpoint ----
        if (
            soma.global_step > 0
            and soma.global_step % int(config.checkpoint_interval) == 0
        ):
            try:
                _save_checkpoint(
                    soma, ckpt_dir, soma.global_step, permanent_every_steps=permanent_every
                )
            except Exception as exc:  # noqa: BLE001 — checkpoint failure is non-fatal
                print(f"train_service: checkpoint save failed: {exc}", file=sys.stderr)

        if max_steps is not None and steps_taken >= max_steps:
            try:
                _save_checkpoint(
                    soma, ckpt_dir, soma.global_step, permanent_every_steps=permanent_every
                )
            except Exception as exc:  # noqa: BLE001
                print(f"train_service: final save failed: {exc}", file=sys.stderr)
            heartbeat.update(step=soma.global_step, status=HEARTBEAT_STATUS_SHUTDOWN)
            return 0


def _build_corpus_blocks(
    lines: list[str], encoder: Any, *, target_block_tokens: int
) -> list[str]:
    """Group adjacent corpus lines into blocks of ~``target_block_tokens`` tokens.

    The feeder requires each text to be at least ``2 * chunk_size`` tokens to
    emit a sample; short-line corpora (like tinyshakespeare) need grouping
    first. We over-shoot the target so every resulting block is long enough
    after tokenizer variation.
    """
    blocks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for line in lines:
        n = len(encoder.tokenize(line))
        buf.append(line)
        buf_tokens += n
        if buf_tokens >= target_block_tokens:
            blocks.append("\n".join(buf))
            buf = []
            buf_tokens = 0
    if buf and buf_tokens >= target_block_tokens // 2:
        blocks.append("\n".join(buf))
    if not blocks:
        # Fallback — join the whole corpus if even the last block was too short.
        blocks = ["\n".join(lines)]
    return blocks


def _install_signal_handlers(shutdown_flag_path: Path) -> None:
    """Translate SIGINT/SIGTERM into a filesystem shutdown signal.

    We prefer the filesystem sentinel so the training loop's existing
    signal-polling branch handles everything in one place.
    """
    import signal as pysignal

    def _handler(signum: int, frame: Any) -> None:
        try:
            shutdown_flag_path.parent.mkdir(parents=True, exist_ok=True)
            shutdown_flag_path.touch()
        except OSError:
            pass

    with contextlib.suppress(ValueError, OSError, AttributeError):
        pysignal.signal(pysignal.SIGINT, _handler)
    with contextlib.suppress(ValueError, OSError, AttributeError):
        pysignal.signal(pysignal.SIGTERM, _handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SOMA training service daemon.")
    parser.add_argument("--config", type=Path, default=Path("configs/current.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on training steps per run (testing only).",
    )
    args = parser.parse_args(argv)

    pidfile = PidFile(Path(".soma-loop/pid/train_service.pid"))
    if check_pid_collision(pidfile.path):
        print(
            f"ERROR: another train_service is already running (pid file: {pidfile.path})",
            file=sys.stderr,
        )
        return 1
    pidfile.write()

    state_dir = Path(".soma-loop/state")
    signals_dir = Path(".soma-loop/signals")
    metrics_path = Path(".soma-loop/metrics/metrics.current.jsonl")
    heartbeat_path = state_dir / "train_heartbeat.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    signals_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    _install_signal_handlers(signals_dir / "shutdown")

    try:
        # Heavy deps are imported lazily so --help and PID guard work cheaply.
        import torch  # noqa: F401

        from soma.core.config import SOMAConfig
        from soma.io.dataset_feeders import TextDatasetFeeder
        from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
        from soma.system import SOMA

        config_path = args.config if args.config.exists() else Path("configs/default.yaml")
        config = SOMAConfig.from_yaml(config_path)
        device = _resolve_device(args.device)

        # Corpus: the bootstrap-produced heldout is small; prefer tinyshakespeare
        # when available for a richer training signal. Individual lines of tiny
        # shakespeare are too short (max ~23 tokens) for a 16+16 chunk split, so
        # we stitch adjacent lines into blocks of roughly ``chunk_size*4`` tokens
        # before handing them to the feeder.
        corpus_path = Path("data/tinyshakespeare.txt")
        if not corpus_path.exists():
            corpus_path = Path("data/heldout.txt")
        corpus_text = corpus_path.read_text(encoding="utf-8")
        corpus_lines = [ln.strip() for ln in corpus_text.splitlines() if ln.strip()]
        if not corpus_lines:
            print(f"ERROR: corpus {corpus_path} has no lines", file=sys.stderr)
            return 1

        tokenizer = train_bpe_tokenizer(corpus_lines, vocab_size=config.vocab_size)
        encoder = TextEncoder(
            tokenizer,
            embed_dim=config.text_embed_dim,
            max_seq_len=config.max_input_tokens,
            device=device,
        )
        chunk_size = max(4, min(16, config.max_input_tokens // 2))
        corpus_blocks = _build_corpus_blocks(
            corpus_lines, encoder, target_block_tokens=chunk_size * 4
        )
        feeder = TextDatasetFeeder(encoder, corpus_blocks, chunk_size=chunk_size)

        soma = SOMA(config, device=device)

        ckpt_dir = args.checkpoint_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        load_result = load_checkpoint_chain(
            current=ckpt_dir / "current.pt",
            last_good=ckpt_dir / "last_good.pt",
            fresh_flag=state_dir / "fresh_init.flag",
        )
        fresh_init = load_result.source == "fresh"
        if not fresh_init:
            load_path = (
                ckpt_dir / "current.pt"
                if load_result.source == "current"
                else ckpt_dir / "last_good.pt"
            )
            try:
                soma.load_state(load_path)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"train_service: SOMA.load_state failed ({exc}); starting fresh",
                    file=sys.stderr,
                )
                fresh_init = True
                (state_dir / "fresh_init.flag").touch()

        heartbeat = Heartbeat(heartbeat_path, device=str(device))
        poller = SignalPoller(signals_dir)

        print(
            f"train_service: pid {os.getpid()} device={device} step={soma.global_step} "
            f"load_source={load_result.source}",
            flush=True,
        )

        return run_training_loop(
            soma=soma,
            feeder=feeder,
            config=config,
            heartbeat=heartbeat,
            poller=poller,
            metrics_path=metrics_path,
            ckpt_dir=ckpt_dir,
            state_dir=state_dir,
            fresh_init=fresh_init,
            max_steps=args.max_steps,
        )
    finally:
        pidfile.remove()


if __name__ == "__main__":
    sys.exit(main())
