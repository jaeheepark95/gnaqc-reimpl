"""Visualize evaluation results as per-category tables.

Reads an `eval_results.csv` produced by `gnaqc.evaluate` and writes two tables:

- test1: Category-A circuits (12)
- test2: Category-B circuits (13)

Each table is rows=circuits (+ trailing "Average"), columns=method × {Hellinger, PST}.
Circuits that were skipped during evaluation (e.g. too large for the backend)
appear as "-". The Average row uses only circuits that were actually evaluated
for a given method/metric.

Outputs (alongside the input CSV unless --output-dir is given):
    test1_<backend>.csv, test1_<backend>.md
    test2_<backend>.csv, test2_<backend>.md

Usage:
    python -m gnaqc.visualize_eval \\
        --results runs/eval/<RUN>/eval_results.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from gnaqc.train import BENCHMARK_CIRCUITS

logger = logging.getLogger(__name__)

CATEGORY_A = BENCHMARK_CIRCUITS[:12]
CATEGORY_B = BENCHMARK_CIRCUITS[12:]

METHOD_ORDER = ["gnaqc", "sabre"]
METHOD_DISPLAY = {"gnaqc": "GNAQC", "sabre": "SABRE"}
METRICS = ["hellinger", "pst"]
METRIC_LABEL = {"hellinger": "Hellinger", "pst": "PST"}


def _build_table(
    df: pd.DataFrame, circuits: list[str], methods: list[str]
) -> pd.DataFrame:
    """Pivot eval rows for a circuit subset into a circuits × (method, metric) table."""
    sub = df[df["circuit"].isin(circuits)]
    columns = pd.MultiIndex.from_product(
        [methods, [METRIC_LABEL[m] for m in METRICS]],
        names=["method", "metric"],
    )
    table = pd.DataFrame(index=list(circuits), columns=columns, dtype=float)

    for _, row in sub.iterrows():
        if row["method"] not in methods:
            continue
        for metric in METRICS:
            table.loc[row["circuit"], (row["method"], METRIC_LABEL[metric])] = row[metric]

    avg = table.mean(axis=0, skipna=True)
    table.loc["Average", :] = avg.values
    return table


def _format_for_display(table: pd.DataFrame) -> pd.DataFrame:
    """Round to 4 decimals and replace NaN with '-' for human output."""
    formatted = table.round(4).astype(object)
    formatted = formatted.where(table.notna(), "-")
    return formatted


def _flat_headers(columns: pd.MultiIndex) -> list[str]:
    short = {"Hellinger": "Hel", "PST": "PST"}
    return [f"{METHOD_DISPLAY.get(m, m)} ({short[metric]})" for m, metric in columns]


def _to_markdown(table: pd.DataFrame, title: str) -> str:
    formatted = _format_for_display(table)
    headers = ["Circuit", *_flat_headers(formatted.columns)]

    def _row_md(cells: list) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    lines = [f"## {title}", "", _row_md(headers),
             _row_md(["---"] * len(headers))]
    for idx, row in formatted.iterrows():
        lines.append(_row_md([idx, *row.tolist()]))
    return "\n".join(lines) + "\n"


def visualize(results_csv: str, output_dir: str | None = None) -> dict[str, Path]:
    results_path = Path(results_csv)
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    df = pd.read_csv(results_path)
    backend = df["backend"].iloc[0] if "backend" in df.columns and len(df) else "unknown"
    # Column set: fix across both tables so test1 and test2 align even when
    # test2 has no rows (e.g., all Category-B circuits too large for the backend).
    methods_in_file = set(df["method"].unique()) if "method" in df.columns else set()
    methods = [m for m in METHOD_ORDER if m in methods_in_file] or METHOD_ORDER

    out_dir = Path(output_dir) if output_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for tag, circuits in (("test1", CATEGORY_A), ("test2", CATEGORY_B)):
        table = _build_table(df, circuits, methods)
        csv_path = out_dir / f"{tag}_{backend}.csv"
        md_path = out_dir / f"{tag}_{backend}.md"

        _format_for_display(table).to_csv(csv_path)
        md_path.write_text(_to_markdown(table, f"{tag} — {backend} (Category {'A' if tag == 'test1' else 'B'})"))

        written[f"{tag}_csv"] = csv_path
        written[f"{tag}_md"] = md_path

        logger.info("\n%s", _to_markdown(table, f"{tag} ({backend})"))

    return written


def main():
    parser = argparse.ArgumentParser(description="Visualize GNAQC eval results as 2 tables (test1/test2)")
    parser.add_argument("--results", required=True, help="Path to eval_results.csv")
    parser.add_argument("--output-dir", default=None, help="Output dir (defaults to results' parent)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    written = visualize(args.results, args.output_dir)
    for key, path in written.items():
        logger.info("%s -> %s", key, path)


if __name__ == "__main__":
    main()
