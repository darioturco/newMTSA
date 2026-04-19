import pandas as pd
import matplotlib.pyplot as plt

# Load data
#df = pd.read_csv("./Experiments/benchmark_results.csv")
df = pd.read_csv("./Experiments/benchmark_results_original.csv")

# Ensure types
df["solved"] = df["solved"].astype(bool)

# Adjust transitions
df["transitions_adj"] = df["transitions"]
df.loc[~df["solved"], "transitions_adj"] = 2000

# Correct family order
families = ["AT", "BW", "DP", "TA", "TL", "CM"]

# Force RANDOM first
heuristics = list(df["heuristic"].unique())
if "RANDOM" in heuristics:
    heuristics.remove("RANDOM")
    heuristics = ["RANDOM"] + heuristics


# -----------------------------
# Helper function
# -----------------------------
def stacked_bar_with_labels(data, title, ylabel, scale=1, suffix=""):
    plt.figure()

    bottom = pd.Series([0]*len(data), index=data.index)

    for family in families:
        values = data[family] / scale

        bars = plt.bar(data.index, values, bottom=bottom, label=family)

        # Labels inside each segment
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bottom.iloc[i] + val/2,
                    f"{int(val)}{suffix}",
                    ha='center',
                    va='center',
                    fontsize=8
                )

        bottom += values

    # Total labels on top
    totals = data.sum(axis=1) / scale
    for i, total in enumerate(totals):
        plt.text(
            i,
            total + max(totals)*0.01,
            f"{int(total)}{suffix}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    plt.title(title)
    plt.xlabel("Heuristic")
    plt.ylabel(ylabel)
    plt.legend(title="Family")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 1. Solved instances
# -----------------------------
solved_group = (
    df.groupby(["heuristic", "family"])["solved"]
    .sum()
    .unstack(fill_value=0)
    .reindex(index=heuristics, columns=families)
)

stacked_bar_with_labels(
    solved_group,
    "Solved instances per heuristic (stacked by family)",
    "Solved instances"
)


# -----------------------------
# 2. Transitions (in K)
# -----------------------------
trans_group = (
    df.groupby(["heuristic", "family"])["transitions_adj"]
    .sum()
    .unstack(fill_value=0)
    .reindex(index=heuristics, columns=families)
)

stacked_bar_with_labels(
    trans_group,
    "Transitions per heuristic (stacked by family)",
    "Transitions (K)",
    scale=1000,
    suffix="K"
)