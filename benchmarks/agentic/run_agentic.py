"""Orchestrator CLI for the agentic benchmark suite.

Usage::

    python -m benchmarks.agentic.run_agentic \\
        --models sota-max,small-high \\
        --tasks fact_recall,tool_learning \\
        --seeds 3 \\
        --agent baseline \\
        --out-md benchmarks/agentic/reports/baseline.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from benchmarks.agentic.agents import AGENT_REGISTRY
from benchmarks.agentic.agents.baseline import BaselineAgent
from benchmarks.agentic.harness import run
from benchmarks.agentic.metrics import TaskResult, format_comparison_table
from benchmarks.agentic.models import MODELS, get_model
from benchmarks.agentic.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)


def _build_agent(agent_name: str, model_tier: str) -> BaselineAgent:
    """Construct an agent by name + model tier."""
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent {agent_name!r}. "
            f"Available: {sorted(AGENT_REGISTRY)}"
        )
    model_cfg = get_model(model_tier)

    if agent_name == "baseline":
        return BaselineAgent(model_config=model_cfg)

    raise ValueError(f"Agent {agent_name!r} not yet implemented.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the agentic benchmark suite."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS),
        help="Comma-separated model tier names (default: all)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASK_REGISTRY),
        help="Comma-separated task names (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds per (model, task) pair",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="baseline",
        choices=sorted(AGENT_REGISTRY),
        help="Agent implementation to benchmark",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per task run",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=None,
        help="Path to write the markdown report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run matrix without executing",
    )

    args = parser.parse_args(argv)
    model_tiers = [m.strip() for m in args.models.split(",")]
    task_names = [t.strip() for t in args.tasks.split(",")]

    # Validate
    for m in model_tiers:
        if m not in MODELS:
            parser.error(
                f"Unknown model {m!r}. Available: {sorted(MODELS)}"
            )
    for t in task_names:
        if t not in TASK_REGISTRY:
            parser.error(
                f"Unknown task {t!r}. "
                f"Available: {sorted(TASK_REGISTRY)}"
            )

    total = len(model_tiers) * len(task_names) * args.seeds
    print(
        f"Benchmark matrix: {len(model_tiers)} models x "
        f"{len(task_names)} tasks x {args.seeds} seeds = {total} runs"
    )

    if args.dry_run:
        for m in model_tiers:
            cfg = get_model(m)
            for t in task_names:
                for s in range(args.seeds):
                    print(
                        f"  {cfg.display_name} | {t} | seed={s}"
                    )
        return

    # Run the matrix
    # results[model_display][task_name] = averaged TaskResult
    results: dict[str, dict[str, TaskResult]] = {}

    for m in model_tiers:
        cfg = get_model(m)
        model_results: dict[str, TaskResult] = {}

        for t in task_names:
            seed_results: list[TaskResult] = []

            for s in range(args.seeds):
                agent = _build_agent(args.agent, m)
                task_cls = TASK_REGISTRY[t]
                task = task_cls(seed=s)

                # Copy tools if the task has them
                if hasattr(task, "TOOLS"):
                    agent.tools = task.TOOLS

                print(
                    f"  Running: {cfg.display_name} | "
                    f"{t} | seed={s} ... ",
                    end="",
                    flush=True,
                )
                result = run(
                    agent,
                    task,
                    task_name=t,
                    agent_name=args.agent,
                    model_name=cfg.display_name,
                    seed=s,
                    max_steps=args.max_steps,
                )
                print(
                    f"done ({result.total_steps} steps, "
                    f"{result.task_result.accuracy:.0f}% acc)"
                )
                seed_results.append(result.task_result)

            # Average across seeds
            avg = _average_results(seed_results)
            model_results[t] = avg

        results[cfg.display_name] = model_results

    # Format and output
    table = format_comparison_table(results)
    print("\n" + table)

    if args.out_md:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            f"# Agentic Benchmark: {args.agent}\n\n{table}\n",
            encoding="utf-8",
        )
        print(f"\nReport written to {out_path}")


def _average_results(results: list[TaskResult]) -> TaskResult:
    """Average TaskResult fields across seeds."""
    if not results:
        return TaskResult()
    n = len(results)
    return TaskResult(
        completion=all(r.completion for r in results),
        accuracy=sum(r.accuracy for r in results) / n,
        steps=round(sum(r.steps for r in results) / n),
        tool_errors=round(sum(r.tool_errors for r in results) / n),
        wall_clock_s=sum(r.wall_clock_s for r in results) / n,
        peak_vram_mb=max(r.peak_vram_mb for r in results),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
