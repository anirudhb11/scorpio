"""Ranking bias^2, variance and ranking error against inference cost.

    python -m scorpio.cli.plot runs/*/metrics_*.json --out biasvar.pdf

One row per (model, dataset), one curve per scorer, one point per number of
rounds.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt

METRICS = [("bias2", "Bias$^2$"), ("variance", "Variance"), ("ranking_error", "Ranking error")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="+", help="metrics_*.json files written by report")
    ap.add_argument("--out", default="biasvar.pdf")
    args = ap.parse_args()

    cells = defaultdict(list)
    for path in args.metrics:
        m = json.load(open(path))
        cells[(m["model"], m["dataset"])].append(m)

    rows = sorted(cells)
    fig, axes = plt.subplots(len(rows), len(METRICS), squeeze=False,
                             figsize=(4 * len(METRICS), 3 * len(rows)))
    for row, (model, dataset) in zip(axes, rows):
        for m in sorted(cells[(model, dataset)], key=lambda m: m["scorer"]):
            points = [p for p in m["curve"] if p["rounds"] > 0]
            cost = [p["total_cost_usd"] for p in points]
            for ax, (key, _) in zip(row, METRICS):
                ax.plot(cost, [p[key] for p in points], marker="o", label=m["scorer"])
        for ax, (_, title) in zip(row, METRICS):
            ax.set_title(f"{model} / {dataset}: {title}")
            ax.set_xlabel("Cost per problem (USD)")
        row[0].legend()

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
