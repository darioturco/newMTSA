"""
Bar plot of solved instances per heuristic/agent from DCSBenchmark CSV files.

Each bar is stacked by problem family (one color per family).
Segment labels show solved count for that family; bar top shows total.

Expected CSV format (per-agent files):
    family,n,k,solved,realizable,transitions,states,time_ms

CSV files are resolved relative to <project_root>/results/:
    random_benchmark.csv, ra_benchmark.csv, bfs_benchmark.csv,
    dqn_benchmark.csv,    ppo_benchmark.csv, sac_benchmark.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR  = _PROJECT_ROOT / "results"

# Display order and CSV filenames (relative to _RESULTS_DIR)
AGENTS: list[tuple[str, str]] = [
    ("Random", "random_benchmark.csv"),
    ("RA",     "ra_benchmark.csv"),
    ("BFS",    "bfs_benchmark.csv"),
    ("DQN",    "dqn_benchmark.csv"),
    ("PPO",    "ppo_benchmark.csv"),
    ("SAC",    "sac_benchmark.csv"),
]

_FAMILY_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def _load_solved(csv_path: Path) -> dict[str, int]:
    """Return {family: solved_count} from a benchmark CSV. Empty dict if missing."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    df["solved"] = df["solved"].astype(str).str.lower() == "true"
    return df[df["solved"]].groupby("family").size().to_dict()


def graph_results(output_path: str | None = None) -> None:
    agent_data: dict[str, dict[str, int]] = {}
    all_families: set[str] = set()

    for label, filename in AGENTS:
        data = _load_solved(_RESULTS_DIR / filename)
        agent_data[label] = data
        all_families.update(data.keys())

    if not all_families:
        print("No solved-instance data found. Run DCSBenchmark first.")
        return

    families = sorted(all_families)
    color_map = {f: _FAMILY_COLORS[i % len(_FAMILY_COLORS)] for i, f in enumerate(families)}

    labels = [a for a, _ in AGENTS]
    x      = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 6))
    bottoms = np.zeros(len(labels), dtype=float)

    for family in families:
        values = np.array([agent_data[a].get(family, 0) for a in labels], dtype=float)
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
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Solved instances", fontsize=11)
    ax.set_title("Solved instances per heuristic / agent", fontsize=13)
    ax.set_ylim(0, max(bottoms) * 1.1 if max(bottoms) > 0 else 1)
    ax.legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()

    out = Path(output_path) if output_path else _PROJECT_ROOT / "benchmark_graph.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Graph saved to {out}")
    plt.show()
