"""Scoring dataclasses and reporting utilities for the agentic benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskResult:
    """Result of a single task run."""

    completion: bool = False
    accuracy: float = 0.0  # 0-100
    steps: int = 0
    tool_errors: int = 0
    wall_clock_s: float = 0.0
    peak_vram_mb: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)


def format_comparison_table(
    results: dict[str, dict[str, TaskResult]],
) -> str:
    """Produce a markdown table: model (rows) x task (cols).

    Parameters
    ----------
    results : dict[model_name, dict[task_name, TaskResult]]

    Returns
    -------
    str  Markdown table.
    """
    if not results:
        return "_No results._"

    # Collect all task names in stable order
    all_tasks: list[str] = []
    for task_dict in results.values():
        for t in task_dict:
            if t not in all_tasks:
                all_tasks.append(t)

    # Header
    header = "| Model | " + " | ".join(all_tasks) + " |"
    sep = "|---|" + "|".join(["---"] * len(all_tasks)) + "|"
    rows = [header, sep]

    for model, task_dict in results.items():
        cells: list[str] = []
        for t in all_tasks:
            tr = task_dict.get(t)
            if tr is None:
                cells.append("-")
            else:
                status = "pass" if tr.completion else "fail"
                cells.append(f"{tr.accuracy:.0f}% ({status}, {tr.steps}s)")
        rows.append(f"| {model} | " + " | ".join(cells) + " |")

    return "\n".join(rows)
