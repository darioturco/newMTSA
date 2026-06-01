"""
Bar plot of solved instances per heuristic from DCSBenchmark CSV files.

Sources (relative to python/results/Benchmark/):
  benchmark_results_original.csv  — columns: family,n,k,heuristic,solved,transitions
                                    heuristics: Ready, Random, BFS  (solved=1/0)
  SuperDFS_results.csv            — columns: Instance,N,K,Name,Transitions,Time
                                    all rows = solved instances
  SuperRL_results.csv             — no header: family,n,k,name,transitions,x
                                    all rows = solved instances

Each bar is stacked by problem family (one color per family).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_BENCHMARK_DIR = _PROJECT_ROOT / "python" / "results" / "Benchmark"

AGENTS_ORDER = ["RA", "SuperDFS", "SuperRL"]

_FAMILY_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def _load_original(heuristic: str) -> dict[str, int]:
    path = _BENCHMARK_DIR / "benchmark_results_original.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    mask = (df["heuristic"] == heuristic) & (df["solved"] == 1)
    return df[mask].groupby("family").size().to_dict()


def _load_instance_csv(filename: str) -> dict[str, int]:
    """Load a CSV with columns Instance,N,K,Name,Transitions,Time where Time=-1 = timeout."""
    path = _BENCHMARK_DIR / filename
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df[df["Transitions"] < 15000].groupby("Instance").size().to_dict()


def _load_superdfs() -> dict[str, int]:
    return _load_instance_csv("SuperDFS_results.csv")


def _load_superrl() -> dict[str, int]:
    path = _BENCHMARK_DIR / "SuperRL_results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, header=None,
                     names=["family", "n", "k", "name", "transitions", "time"])
    solved = df[df["time"] != -1][["family", "n", "k"]].drop_duplicates()
    return solved.groupby("family").size().to_dict()


def graph_results(output_path: str | None = None, budget: int = 2500) -> None:
    agent_data: dict[str, dict[str, int]] = {
        "Random":   _load_original("Random"),
        "BFS":      _load_original("BFS"),
        "RA":       _load_instance_csv("RA.csv"),
        "SuperDFS": _load_superdfs(),
        "SuperRL":  _load_superrl(),
    }

    all_families: set[str] = set()
    for data in agent_data.values():
        all_families.update(data.keys())

    if not all_families:
        print("No solved-instance data found.")
        return

    active_agents = [a for a in AGENTS_ORDER if sum(agent_data[a].values()) > 0]
    families      = sorted(all_families)
    color_map     = {f: _FAMILY_COLORS[i % len(_FAMILY_COLORS)] for i, f in enumerate(families)}

    x       = np.arange(len(active_agents))
    bottoms = np.zeros(len(active_agents), dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, len(active_agents) * 1.5), 6))

    for family in families:
        values = np.array([agent_data[a].get(family, 0) for a in active_agents], dtype=float)
        bars   = ax.bar(x, values, bottom=bottoms, color=color_map[family], label=family,
                        edgecolor="white", linewidth=0.5)
        for bar, val, bot in zip(bars, values, bottoms):
            if val >= 1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + val / 2,
                    str(int(val)),
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold",
                )
        bottoms += values

    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(i, total + max(bottoms) * 0.01, str(int(total)),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(active_agents, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Solved instances", fontsize=11)
    ax.set_title(f"Solved instances per heuristic  |  Budget: {budget:,} transitions",
                 fontsize=13)
    ax.set_ylim(0, max(bottoms) * 1.12 if max(bottoms) > 0 else 1)
    ax.legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()

    out = Path(output_path) if output_path else _PROJECT_ROOT / "benchmark_graph.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Graph saved to {out}")
    plt.show()


if __name__ == "__main__":
    graph_results()
