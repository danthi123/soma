"""One-time bootstrap for the autonomous SOMA improvement loop.

Creates .soma-loop/ directory tree, seeds configs/current.yaml from default,
writes data/fixed_prompts.txt, builds data/heldout.txt split and
data/corpus_token_freq.json KL reference.

Idempotent by default; pass --force to regenerate heldout + token_freq
(configs/current.yaml and data/fixed_prompts.txt are NEVER overwritten
once present — per design G64).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

FIXED_PROMPTS: list[str] = [
    "To be, or not to be,",
    "The king said,",
    "O Romeo, Romeo,",
    "What light through",
    "Shall I compare thee",
    "hello",
    "what is your name",
    "tell me about yourself",
    "how do you feel",
    "who are you",
    "The quick brown",
    "Once upon a",
    "In the beginning",
    "Long ago in",
    "The secret of",
    "",
    ".",
    "a",
    "a a a a a a a a a a a a a a a a",
    "Now is the winter of our discontent made glorious summer by this sun of York",
]


def ensure_state_dirs(repo_root: Path) -> None:
    """Create .soma-loop/ structure + initial state files.

    Idempotent — creates missing dirs/files, leaves existing alone.
    """
    dirs = [
        ".soma-loop/signals",
        ".soma-loop/state/ticks",
        ".soma-loop/metrics",
        ".soma-loop/reports/chat",
        ".soma-loop/logs",
        ".soma-loop/pid",
    ]
    for rel in dirs:
        (repo_root / rel).mkdir(parents=True, exist_ok=True)

    empty_files = [
        ".soma-loop/state/change_log.jsonl",
        ".soma-loop/state/approval_queue.jsonl",
    ]
    for rel in empty_files:
        path = repo_root / rel
        if not path.exists():
            path.touch()

    cf_path = repo_root / ".soma-loop/state/consecutive_failures.json"
    if not cf_path.exists():
        cf_path.write_text(
            json.dumps({"count": 0, "last_reset_ts": None, "last_failure_ts": None}),
            encoding="utf-8",
        )


def seed_current_yaml(default: Path, current: Path, *, force: bool) -> None:
    """Copy default.yaml to current.yaml if current doesn't exist.

    Per G64, NEVER overwrites current.yaml even with --force.
    """
    if current.exists():
        return
    current.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(default, current)


def seed_fixed_prompts(path: Path, *, force: bool) -> None:
    """Write the default 20 fixed prompts if file absent. Never overwrite."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(FIXED_PROMPTS) + "\n", encoding="utf-8")


def build_heldout_split(corpus: Path, out: Path, *, ratio: float = 0.1, seed: int = 42) -> None:
    """Deterministic ratio% split of corpus lines into out."""
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(len(lines)), max(1, int(len(lines) * ratio))))
    out.write_text("\n".join(lines[i] for i in selected) + "\n", encoding="utf-8")


def build_token_freq(corpus: Path, out: Path) -> None:
    """Empirical word-frequency dict (JSON) over corpus — used as KL reference.

    Simple whitespace tokenization; the true BPE-tokenized distribution is
    computed by test_harness.py when available. This is the pre-BPE fallback.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    counter: Counter[str] = Counter()
    for line in corpus.read_text(encoding="utf-8").splitlines():
        counter.update(line.split())
    out.write_text(json.dumps(dict(counter)), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap SOMA autonomous loop.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate heldout and token_freq (config + fixed_prompts always preserved).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/tinyshakespeare.txt"),
        help="Corpus file for heldout split and token-freq generation.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repo root (default: current directory).",
    )
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    default_yaml = repo / "configs/default.yaml"
    current_yaml = repo / "configs/current.yaml"
    prompts = repo / "data/fixed_prompts.txt"
    heldout = repo / "data/heldout.txt"
    token_freq = repo / "data/corpus_token_freq.json"

    if not default_yaml.exists():
        print(f"ERROR: {default_yaml} missing", file=sys.stderr)
        return 1
    if not args.corpus.exists():
        print(f"ERROR: corpus {args.corpus} missing", file=sys.stderr)
        return 1

    ensure_state_dirs(repo)
    seed_current_yaml(default_yaml, current_yaml, force=args.force)
    seed_fixed_prompts(prompts, force=args.force)

    if args.force or not heldout.exists():
        build_heldout_split(args.corpus, heldout, ratio=0.1, seed=42)
    if args.force or not token_freq.exists():
        build_token_freq(args.corpus, token_freq)

    print("Bootstrap complete. Next steps:")
    print("  1. Review data/fixed_prompts.txt (edit if desired)")
    print("  2. Invoke soma-bootstrap-kb skill (once, manually)")
    print("  3. Start train_service: python scripts/train_service.py")
    print("  4. Schedule run_tick.ps1 and auto_revert.py in Task Scheduler")
    return 0


if __name__ == "__main__":
    sys.exit(main())
