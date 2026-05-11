"""Report generation helper."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report_stub(metrics: MetricsReport) -> str:
    """Return a minimal report stub.
    """
    header = "| Scenario | Expected route | Actual route | Success | Retries | Interrupts |"
    sep = "|---|---|---|---:|---:|---:|"
    rows = [
        f"| {item.scenario_id} | {item.expected_route} | {item.actual_route or ''} | {str(item.success).lower()} | {item.retry_count} | {item.interrupt_count} |"
        for item in metrics.scenario_metrics
    ]
    table = "\n".join([header, sep, *rows]
                      ) if rows else "(no scenarios recorded)"
    return f"""# Day 08 Lab Report

## 1. Team / student

- Name:
- Repo/commit:
- Date:

## 2. Architecture

Describe your graph nodes, edges, state fields, and reducers.

## 3. State schema

List important fields and whether they are overwrite or append-only.

| Field | Reducer | Why |
|---|---|---|
| messages | append | audit conversation/events |
| route | overwrite | current route only |

## 4. Scenario results

{table}

Summary:

- Total scenarios: {metrics.total_scenarios}
- Success rate: {metrics.success_rate:.2%}
- Average nodes visited: {metrics.avg_nodes_visited:.2f}
- Total retries: {metrics.total_retries}
- Total interrupts: {metrics.total_interrupts}

## 5. Failure analysis

Describe at least two failure modes you considered:

1. Retry or tool failure:
2. Risky action without approval:

## 6. Persistence / recovery evidence

Explain how you used checkpointer, thread id, state history, or crash-resume.

## 7. Extension work

Describe any extension you completed: SQLite/Postgres, time travel, fan-out/fan-in, graph diagram, tracing.

## 8. Improvement plan

If you had one more day, what would you productionize first?
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report_stub(metrics), encoding="utf-8")
