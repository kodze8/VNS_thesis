
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import defaultdict

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

RESULTS_DIR  = "../results"
PLOTS_DIR    = "plots"

# the two summary files produced by run_tuning.py and run_test.py
TUNING_SUMMARY = "../results/tuning_summary.csv"
TEST_SUMMARY   = "../results/test_summary.csv"

KNOWN_OPTIMALS = {
    # tuning
    "eil51":   426,
    "eil76":   538,
    "rat99":   1211,
    "eil101":  629,
    "ch150":   6528,
    "kroA100": 21282,
    "kroA150": 26524,
    # test
    "berlin52": 7542,
    "kroB100":  22141,
    "kroA200":  29368,
    "lin318":   41345,
}

TUNING_INSTANCES = ["eil51", "eil76", "rat99", "eil101", "ch150", "kroA100", "kroA150"]
TEST_INSTANCES   = ["berlin52", "kroB100", "kroA200", "lin318"]

# display order (small -> large) for plots/tables
TUNING_ORDER = ["eil51", "eil76", "rat99", "eil101", "kroA100", "ch150", "kroA150"]
TEST_ORDER   = ["berlin52", "kroB100", "kroA200", "lin318"]


NEIGHBORHOOD_ORDER = ["or_opt_1", "or_opt_2", "or_opt_3",
                      "double_bridge", "double_double_bridge"]

NB_LABELS = {
    "or_opt_1": "N1\nOr-opt 1",
    "or_opt_2": "N2\nOr-opt 2",
    "or_opt_3": "N3\nOr-opt 3",
    "double_bridge": "N4\nDouble bridge",
    "double_double_bridge": "N5\nDouble db.",
}


REFERENCE_CONFIG = {"k_max": 5, "max_no_improve": 100, "init_method": "nearest_neighbor"}

COL_RED  = "#BC0032"
COL_BLUE = "#2166ac"
COL_GREEN = "#33a02c"
THREE_COLORS = [COL_BLUE, COL_RED, COL_GREEN]

os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────

def load_all_runs(instance_filter=None) -> pd.DataFrame:
    """Load every per-run log CSV, optionally restricted to a set of instances."""
    dfs = []
    skip = {"tuning_summary.csv", "test_summary.csv", "experiment_summary.csv"}
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(".csv") and file not in skip:
            df = pd.read_csv(os.path.join(RESULTS_DIR, file))
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No run CSVs found in results/.")
    combined = pd.concat(dfs, ignore_index=True)
    if instance_filter is not None:
        combined = combined[combined["instance_name"].isin(instance_filter)]
    print(f"Loaded {len(dfs)} run files | {len(combined)} log entries"
          + (f" (filtered to {len(instance_filter)} instances)" if instance_filter else ""))
    return combined


def load_summary(path) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)


# ─────────────────────────────────────────────
#  Config filtering helpers
# ─────────────────────────────────────────────

def filter_config(df, k_max, max_no_improve, init_method):
    return df[(df["k_max"] == k_max) &
              (df["max_no_improve"] == max_no_improve) &
              (df["init_method"] == init_method)]


def reference(df):
    """Rows for the reference configuration (C3)."""
    return filter_config(df, **REFERENCE_CONFIG)


# ─────────────────────────────────────────────
#  Data-driven configuration selection (RQ1 -> RQ2 bridge)
# ─────────────────────────────────────────────

def select_best_config(tuning_summary: pd.DataFrame):
    """
    Rank the six configurations by mean gap across the tuning instances and
    return the best one. This replaces the old hardcoded C3 assumption: the
    configuration carried to the test set is chosen from the data.
    """
    print("\n" + "=" * 55)
    print("  Configuration selection on tuning set")
    print("=" * 55)

    grp = (tuning_summary
           .groupby(["k_max", "max_no_improve", "init_method"])
           .agg(mean_gap=("gap", "mean"),
                std_gap=("gap", "std"),
                mean_time=("time", "mean"))
           .round(4)
           .sort_values("mean_gap"))

    print("\nConfigurations ranked by mean gap (lower is better):")
    print(grp.to_string())

    best = grp.index[0]
    chosen = {"k_max": int(best[0]),
              "max_no_improve": int(best[1]),
              "init_method": best[2]}
    print(f"\n  Selected configuration: {chosen}")
    return chosen


# ─────────────────────────────────────────────
#  4.1 — Solution Quality (reference config, tuning set)
# ─────────────────────────────────────────────

def section_4_1(summary: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  4.1 Solution Quality Across Tuning Instances")
    print("=" * 55)

    ref = reference(summary)
    table = ref.groupby("instance").agg(
        optimal=("optimal", "first"),
        best_cost=("best_cost", "min"),
        mean_cost=("best_cost", "mean"),
        std_cost=("best_cost", "std"),
        mean_gap=("gap", "mean"),
        min_gap=("gap", "min"),
        mean_time=("time", "mean"),
    ).round(4)
    table = table.reindex([i for i in TUNING_ORDER if i in table.index])
    print("\nTable 4.1 — Solution quality (reference config):")
    print(table.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(table.index.tolist(), table["mean_gap"].values,
           color=COL_RED, alpha=0.8, zorder=3)
    ax.set_xlabel("Instance")
    ax.set_ylabel("Mean Gap from Optimal (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/4_1_solution_quality.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/4_1_solution_quality.png")


# ─────────────────────────────────────────────
#  4.2 — Effect of k_max
# ─────────────────────────────────────────────

def section_4_2(summary: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  4.2 Effect of k_max")
    print("=" * 55)

    study = summary[(summary["init_method"] == "nearest_neighbor") &
                    (summary["max_no_improve"] == 100)]

    table = study.groupby(["instance", "k_max"]).agg(
        mean_gap=("gap", "mean"), std_gap=("gap", "std")).round(4)
    print("\nTable 4.2 — Mean gap by k_max:")
    print(table.to_string())

    instances = [i for i in TUNING_ORDER if i in study["instance"].unique()]
    k_vals = sorted(study["k_max"].unique())
    x = np.arange(len(instances))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, k in enumerate(k_vals):
        vals = [study[(study["instance"] == inst) & (study["k_max"] == k)]["gap"].mean()
                for inst in instances]
        ax.bar(x + idx * width, vals, width, label=f"k_max={k}",
               color=THREE_COLORS[idx % 3], alpha=0.85, zorder=3)
    ax.set_xticks(x + width)
    ax.set_xticklabels(instances)
    ax.set_xlabel("Instance")
    ax.set_ylabel("Mean Gap from Optimal (%)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/4_2_kmax_effect.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/4_2_kmax_effect.png")


# ─────────────────────────────────────────────
#  4.3 — Effect of Initialization Method
# ─────────────────────────────────────────────

def section_4_3(runs: pd.DataFrame, summary: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  4.3 Effect of Initialization Method")
    print("=" * 55)

    study = summary[(summary["k_max"] == 5) & (summary["max_no_improve"] == 100)]
    table = study.groupby(["instance", "init_method"]).agg(
        mean_gap=("gap", "mean"), std_gap=("gap", "std"),
        mean_time=("time", "mean")).round(4)
    print("\nTable 4.3 — Mean gap by init method:")
    print(table.to_string())

    # convergence curves for two representative tuning instances
    show = [i for i in ["eil51", "eil101"] if i in runs["instance_name"].unique()]
    fig, axes = plt.subplots(1, len(show), figsize=(6.5 * len(show), 5))
    if len(show) == 1:
        axes = [axes]
    for ax, instance in zip(axes, show):
        for init, color, label in [("nearest_neighbor", COL_RED, "Nearest Neighbor"),
                                    ("random", COL_BLUE, "Random")]:
            sub = filter_config(runs[runs["instance_name"] == instance], 5, 100, init)
            if sub.empty:
                continue
            avg = sub.groupby("iteration")["best_cost"].mean()
            ax.plot(avg.index, avg.values, color=color, label=label, linewidth=1.8)
        opt = KNOWN_OPTIMALS.get(instance)
        if opt:
            ax.axhline(opt, color="black", linestyle="--", linewidth=1, label="Optimal")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Cost")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/4_3_init_convergence.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/4_3_init_convergence.png")


# ─────────────────────────────────────────────
#  4.4 — Effect of Stopping Criterion
# ─────────────────────────────────────────────

def section_4_4(runs: pd.DataFrame, summary: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  4.4 Effect of Stopping Criterion")
    print("=" * 55)

    study = summary[(summary["k_max"] == 5) &
                    (summary["init_method"] == "nearest_neighbor")]
    table_a = study.groupby(["instance", "max_no_improve"]).agg(
        mean_gap=("gap", "mean"), std_gap=("gap", "std"),
        mean_iters=("total_iters", "mean")).round(4)
    print("\nTable 4.4a — Mean gap by max_no_improve:")
    print(table_a.to_string())

    # convergence point analysis on reference config
    best = reference(runs)
    results = []
    for (instance, seed), group in best.groupby(["instance_name", "seed"]):
        group = group.sort_values("iteration")
        bc = group["best_cost"]
        last_improve = group.loc[bc != bc.shift(), "iteration"].max()
        total_iters = group["iteration"].max()
        tvals = group.loc[group["iteration"] == last_improve, "time_elapsed"].values
        results.append({
            "instance": instance, "seed": seed,
            "last_improve_iter": last_improve,
            "total_iters": total_iters,
            "wasted_iters": total_iters - last_improve,
            "time_at_converge": tvals[0] if len(tvals) else None,
        })
    df = pd.DataFrame(results)
    table_b = df.groupby("instance").agg(
        mean_last_improve=("last_improve_iter", "mean"),
        mean_total_iters=("total_iters", "mean"),
        mean_wasted=("wasted_iters", "mean"),
        mean_time_converge=("time_at_converge", "mean")).round(2)
    table_b = table_b.reindex([i for i in TUNING_ORDER if i in table_b.index])
    print("\nTable 4.4b — Convergence point analysis:")
    print(table_b.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ni_vals = sorted(study["max_no_improve"].unique())
    inst_list = [i for i in TUNING_ORDER if i in study["instance"].unique()]
    x = np.arange(len(inst_list))
    width = 0.25
    for idx, ni in enumerate(ni_vals):
        vals = [study[(study["instance"] == inst) &
                      (study["max_no_improve"] == ni)]["gap"].mean()
                for inst in inst_list]
        axes[0].bar(x + idx * width, vals, width, label=f"no_improve={ni}",
                    color=THREE_COLORS[idx % 3], alpha=0.85, zorder=3)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(inst_list, rotation=15)
    axes[0].set_xlabel("Instance")
    axes[0].set_ylabel("Mean Gap (%)")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    instances = table_b.index.tolist()
    x2 = np.arange(len(instances))
    w2 = 0.35
    axes[1].bar(x2 - w2/2, table_b["mean_last_improve"], w2,
                label="Last Improvement", color=COL_RED, alpha=0.85, zorder=3)
    axes[1].bar(x2 + w2/2, table_b["mean_total_iters"], w2,
                label="Total Iterations", color=COL_BLUE, alpha=0.85, zorder=3)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(instances, rotation=15)
    axes[1].set_xlabel("Instance")
    axes[1].set_ylabel("Iterations")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/4_4_stopping_criterion.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/4_4_stopping_criterion.png")


# ─────────────────────────────────────────────
#  4.5 — Robustness Across Seeds
# ─────────────────────────────────────────────

def section_4_5(summary: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  4.5 Robustness Across Seeds")
    print("=" * 55)

    ref = reference(summary)
    table = ref.groupby("instance").agg(
        mean_gap=("gap", "mean"), std_gap=("gap", "std"),
        min_gap=("gap", "min"), max_gap=("gap", "max")).round(4)
    table = table.reindex([i for i in TUNING_ORDER if i in table.index])
    print("\nTable 4.5 — Gap statistics across seeds:")
    print(table.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(table.index.tolist(), table["mean_gap"].values,
                yerr=table["std_gap"].values, fmt="o", color=COL_RED,
                ecolor=COL_BLUE, elinewidth=2, capsize=5, markersize=7, zorder=3)
    ax.set_xlabel("Instance")
    ax.set_ylabel("Gap from Optimal (%)")
    ax.grid(linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/4_5_robustness.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/4_5_robustness.png")


# ─────────────────────────────────────────────
#  TEST / VALIDATION chapter — chosen config on held-out instances
# ─────────────────────────────────────────────

def section_test(test_summary: pd.DataFrame, chosen: dict):
    print("\n" + "=" * 55)
    print("  Held-out Test Results (chosen configuration)")
    print("=" * 55)

    sub = filter_config(test_summary, **chosen)
    if sub.empty:
        print("  No test rows match the chosen config. Did you run run_test.py "
              "with this config?")
        return

    table = sub.groupby("instance").agg(
        optimal=("optimal", "first"),
        best_cost=("best_cost", "min"),
        mean_cost=("best_cost", "mean"),
        std_cost=("best_cost", "std"),
        mean_gap=("gap", "mean"),
        min_gap=("gap", "min"),
        max_gap=("gap", "max"),
        mean_time=("time", "mean"),
    ).round(4)
    table = table.reindex([i for i in TEST_ORDER if i in table.index])
    print("\nTest set — chosen configuration:")
    print(table.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(table.index.tolist(), table["mean_gap"].values,
           color=COL_RED, alpha=0.85, zorder=3)
    ax.set_xlabel("Test Instance")
    ax.set_ylabel("Mean Gap from Optimal (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/test_results.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/test_results.png")


# ─────────────────────────────────────────────
#  5.1 — Search Trajectory
# ─────────────────────────────────────────────

def section_5_1(runs: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  5.1 Search Trajectory Analysis")
    print("=" * 55)

    bc = reference(runs)
    instances = [i for i in TUNING_ORDER if i in bc["instance_name"].unique()]
    ncols = 4
    nrows = int(np.ceil(len(instances) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for idx, instance in enumerate(instances):
        ax = axes[idx]
        sub = bc[bc["instance_name"] == instance]
        for seed, group in sub.groupby("seed"):
            group = group.sort_values("iteration")
            ax.plot(group["iteration"], group["best_cost"],
                    alpha=0.4, linewidth=1, color=COL_RED)
        avg = sub.groupby("iteration")["best_cost"].mean()
        ax.plot(avg.index, avg.values, color="black", linewidth=2, label="Mean")
        opt = KNOWN_OPTIMALS.get(instance)
        if opt:
            ax.axhline(opt, color=COL_BLUE, linestyle="--", linewidth=1.2, label="Optimal")
        # instance name kept as small in-axes label (sub-plot identifier, not a title)
        ax.set_title(instance, fontsize=9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Cost")
        ax.legend(fontsize=7)
        ax.grid(linestyle="--", alpha=0.4)
    for idx in range(len(instances), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/5_1_convergence_curves.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/5_1_convergence_curves.png")


# ─────────────────────────────────────────────
#  5.2 — Neighborhood Contribution
# ─────────────────────────────────────────────

def section_5_2(runs: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  5.2 Neighborhood Contribution Analysis")
    print("=" * 55)

    bc = reference(runs)
    rows = []
    for nb in NEIGHBORHOOD_ORDER:
        sub = bc[bc["neighborhood"] == nb]
        used = len(sub)
        improved = int(sub["improved"].sum())
        rate = round(improved / used * 100, 2) if used else 0
        rows.append({"neighborhood": nb, "times_used": used,
                     "improvements": improved, "contribution_rate": rate})
    table = pd.DataFrame(rows)
    print("\nTable 5.2 — Neighborhood contribution:")
    print(table.to_string(index=False))

    labels = [NB_LABELS[n] for n in table["neighborhood"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(labels, table["times_used"], color=COL_BLUE, alpha=0.85, zorder=3)
    axes[0].set_xlabel("Neighborhood")
    axes[0].set_ylabel("Times Used")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    axes[1].bar(labels, table["contribution_rate"], color=COL_RED, alpha=0.85, zorder=3)
    axes[1].set_xlabel("Neighborhood")
    axes[1].set_ylabel("Contribution Rate (%)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/5_2_neighborhood_contribution.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/5_2_neighborhood_contribution.png")


# ─────────────────────────────────────────────
#  5.3 — Shaking Necessity (+ recovery ratio per k)  [EXTENDED]
# ─────────────────────────────────────────────

def section_5_3(runs: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  5.3 Shaking Necessity Analysis")
    print("=" * 55)

    bc = reference(runs)
    table = bc.groupby("neighborhood").agg(
        mean_shake_delta=("shake_delta", "mean"),
        mean_ls_recovery=("ls_recovery", "mean"),
        improvement_rate=("improved", "mean")).round(4)
    table = table.reindex([n for n in NEIGHBORHOOD_ORDER if n in table.index])
    table["improvement_rate"] = (table["improvement_rate"] * 100).round(2)

    # NEW: recovery ratio = total ls_recovery / total shake_delta per neighborhood
    ratio = bc.groupby("neighborhood").apply(
        lambda g: round(g["ls_recovery"].sum() / g["shake_delta"].sum(), 4)
        if g["shake_delta"].sum() != 0 else np.nan,
        include_groups=False)
    table["recovery_ratio"] = ratio.reindex(table.index)

    print("\nTable 5.3 — Shake disruption, LS recovery, and recovery ratio:")
    print(table.to_string())

    labels = [NB_LABELS[n] for n in table.index]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(table))
    width = 0.35
    axes[0].bar(x - width/2, table["mean_shake_delta"], width,
                label="Shake Delta", color=COL_RED, alpha=0.85, zorder=3)
    axes[0].bar(x + width/2, table["mean_ls_recovery"], width,
                label="LS Recovery", color=COL_BLUE, alpha=0.85, zorder=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Cost Change")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    axes[1].scatter(bc["shake_delta"], bc["ls_recovery"],
                    alpha=0.1, s=5, color=COL_RED)
    axes[1].set_xlabel("Shake Delta (disruption)")
    axes[1].set_ylabel("LS Recovery")
    axes[1].grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/5_3_shaking_analysis.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/5_3_shaking_analysis.png")


# ─────────────────────────────────────────────
#  5.4 — Edge Frequency
# ─────────────────────────────────────────────

def section_5_4(runs: pd.DataFrame, instance_name="eil101"):
    print("\n" + "=" * 55)
    print(f"  5.4 Edge Frequency Analysis — {instance_name}")
    print("=" * 55)

    sub = reference(runs[runs["instance_name"] == instance_name])
    if sub.empty:
        print(f"  No data for {instance_name}. Skipping.")
        return

    best_tours = sub.groupby("seed")["best_tour"].last()
    n = None
    edge_counts = defaultdict(int)
    total = 0
    for tour_str in best_tours:
        try:
            tour = ast.literal_eval(tour_str)
        except Exception:
            continue
        if n is None:
            n = len(tour)
        total += 1
        for i in range(len(tour)):
            a, b = tour[i], tour[(i + 1) % len(tour)]
            edge_counts[(min(a, b), max(a, b))] += 1
    if not edge_counts:
        print("  Could not parse tours. Skipping.")
        return

    freq = np.zeros((n, n))
    for (a, b), c in edge_counts.items():
        freq[a][b] = freq[b][a] = c / total
    print(f"  {total} tours | {n} cities")
    print(f"  Edges in ALL runs     : {sum(1 for c in edge_counts.values() if c == total)}")
    print(f"  Edges in >=80% of runs: {sum(1 for c in edge_counts.values() if c/total >= 0.8)}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(freq, ax=ax, cmap="Reds", xticklabels=False, yticklabels=False,
                cbar_kws={"label": "Edge Frequency (fraction of runs)"})
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/5_4_edge_frequency_{instance_name}.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/5_4_edge_frequency_{instance_name}.png")


# ─────────────────────────────────────────────
#  5.5 — Single Run Walkthrough
# ─────────────────────────────────────────────

def section_5_5(runs: pd.DataFrame, instance_name="eil51", seed=42):
    print("\n" + "=" * 55)
    print(f"  5.5 Single Run Walkthrough — {instance_name} seed={seed}")
    print("=" * 55)

    sub = reference(runs[(runs["instance_name"] == instance_name) &
                         (runs["seed"] == seed)]).sort_values("iteration")
    if sub.empty:
        print("  No data. Skipping.")
        return

    improvements = sub[sub["improved"] == True]
    print(f"  Total iterations : {len(sub)} | Improvements : {len(improvements)}")
    print(f"\n  {'Iter':>5} | {'k':>3} | {'Neighborhood':<22} | "
          f"{'Before':>10} | {'After':>10} | {'Best':>10}")
    print(f"  {'-'*74}")
    for _, r in improvements.iterrows():
        print(f"  {int(r['iteration']):>5} | {int(r['k']):>3} | {r['neighborhood']:<22} | "
              f"{r['cost_before']:>10.2f} | {r['cost_after']:>10.2f} | {r['best_cost']:>10.2f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sub["iteration"], sub["best_cost"], color=COL_RED,
            linewidth=1.8, zorder=3, label="Best Cost")
    colors_k = {1: COL_GREEN, 2: "#ff7f00", 3: "#6a3d9a", 4: COL_BLUE, 5: "#e31a1c"}
    for _, r in improvements.iterrows():
        k = int(r["k"])
        ax.axvline(r["iteration"], color=colors_k.get(k, "gray"), alpha=0.6, linewidth=1.2)
    opt = KNOWN_OPTIMALS.get(instance_name)
    if opt:
        ax.axhline(opt, color="black", linestyle="--", linewidth=1.2,
                   label=f"Optimal ({opt})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cost")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/5_5_single_run_{instance_name}_seed{seed}.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/5_5_single_run_{instance_name}_seed{seed}.png")


# ─────────────────────────────────────────────
#  5.6 — Escalation behaviour  [NEW INSIGHT]
# ─────────────────────────────────────────────

def section_5_6(runs: pd.DataFrame):
    """
    How the search uses the hierarchy:
      (a) how often each k is reached (used) and how often it produces an
          improvement, and
      (b) whether the k that produces improvements rises over the course of a
          run (evidence that the escalation mechanism is genuinely used).
    Uses only logged fields (k, improved, iteration).
    """
    print("\n" + "=" * 55)
    print("  5.6 Escalation Behaviour")
    print("=" * 55)

    bc = reference(runs)

    # (a) usage and improvements per k
    usage = bc["k"].value_counts().sort_index()
    imp = bc[bc["improved"] == True]
    imp_by_k = imp["k"].value_counts().sort_index()
    esc = pd.DataFrame({
        "times_used": usage,
        "improvements": imp_by_k
    }).fillna(0).astype(int)
    esc["share_of_improvements_%"] = (
        esc["improvements"] / esc["improvements"].sum() * 100).round(2)
    esc.index.name = "k"
    print("\nTable 5.6a — How often each k is reached and how often it improves:")
    print(esc.to_string())

    # (b) winning-k over the run, by quartile of iteration (per instance, pooled)
    print("\nTable 5.6b — Mean k of IMPROVING steps by run phase (per instance):")
    phase_rows = []
    for instance, g in imp.groupby("instance_name"):
        g = g.copy()
        if g["iteration"].nunique() < 4:
            continue
        try:
            g["phase"] = pd.qcut(g["iteration"], 4,
                                 labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except ValueError:
            continue
        means = g.groupby("phase", observed=True)["k"].mean()
        row = {"instance": instance}
        row.update({p: round(means.get(p, np.nan), 3) for p in ["Q1", "Q2", "Q3", "Q4"]})
        phase_rows.append(row)
    phase_df = pd.DataFrame(phase_rows).set_index("instance")
    phase_df = phase_df.reindex([i for i in TUNING_ORDER if i in phase_df.index])
    print(phase_df.to_string())

    # plot: left = improvements per k; right = mean winning-k by phase, pooled
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [NB_LABELS.get(NEIGHBORHOOD_ORDER[k-1], str(k)) for k in esc.index]
    axes[0].bar(labels, esc["improvements"], color=COL_RED, alpha=0.85, zorder=3)
    axes[0].set_xlabel("Neighborhood (k)")
    axes[0].set_ylabel("Number of Improvements")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    pooled = imp.copy()
    if pooled["iteration"].nunique() >= 4:
        pooled["phase"] = pd.qcut(pooled["iteration"], 4,
                                  labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        pooled_means = pooled.groupby("phase", observed=True)["k"].mean()
        axes[1].plot(pooled_means.index.astype(str), pooled_means.values,
                     marker="o", color=COL_BLUE, linewidth=2, zorder=3)
        axes[1].set_xlabel("Run Phase (iteration quartile)")
        axes[1].set_ylabel("Mean k of Improving Steps")
        axes[1].grid(linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/5_6_escalation.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/5_6_escalation.png")


# ─────────────────────────────────────────────
#  5.7 — Improvement timing on large instances  [NEW INSIGHT]
# ─────────────────────────────────────────────

def section_5_7(runs: pd.DataFrame):
    """
    For each instance, the time at which improvements occur. On large,
    time-limited instances improvements should keep arriving up to the limit
    (search still progressing), whereas small instances stop improving early.
    Uses logged time_elapsed and improved.
    """
    print("\n" + "=" * 55)
    print("  5.7 Improvement Timing")
    print("=" * 55)

    bc = reference(runs)
    rows = []
    for instance, g in bc.groupby("instance_name"):
        imp = g[g["improved"] == True]
        if imp.empty:
            continue
        last_imp_time = imp["time_elapsed"].max()
        total_time = g["time_elapsed"].max()
        rows.append({
            "instance": instance,
            "last_improve_time": round(last_imp_time, 2),
            "total_time": round(total_time, 2),
            "fraction_of_run": round(last_imp_time / total_time, 3) if total_time else np.nan,
        })
    tdf = pd.DataFrame(rows).set_index("instance")
    tdf = tdf.reindex([i for i in TUNING_ORDER if i in tdf.index])
    print("\nTable 5.7 — When the last improvement occurs (time):")
    print(tdf.to_string())
    print("  (fraction_of_run near 1.0 => still improving at the limit; "
          "near 0 => converged early)")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(tdf.index.tolist(), tdf["fraction_of_run"].values,
           color=COL_RED, alpha=0.85, zorder=3)
    ax.set_xlabel("Instance")
    ax.set_ylabel("Last Improvement Time / Total Time")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/5_7_improvement_timing.png", dpi=150)
    plt.close()
    print(f"  -> {PLOTS_DIR}/5_7_improvement_timing.png")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\nLoading data...")

    # tuning analysis
    tuning_runs = load_all_runs(instance_filter=TUNING_INSTANCES)
    tuning_summary = load_summary(TUNING_SUMMARY)

    section_4_1(tuning_summary)
    section_4_2(tuning_summary)
    section_4_3(tuning_runs, tuning_summary)
    section_4_4(tuning_runs, tuning_summary)
    section_4_5(tuning_summary)

    # choose best config from tuning data, then evaluate the test set
    chosen = select_best_config(tuning_summary)
    try:
        test_summary = load_summary(TEST_SUMMARY)
        section_test(test_summary, chosen)
    except FileNotFoundError:
        print("\n[info] test_summary.csv not found yet — run run_test.py with the "
              "chosen config, then re-run for the validation section.")

    # explainability (reference config, tuning instances)
    section_5_1(tuning_runs)
    section_5_2(tuning_runs)
    section_5_3(tuning_runs)
    section_5_4(tuning_runs, instance_name="eil101")
    section_5_5(tuning_runs, instance_name="eil51", seed=42)
    section_5_6(tuning_runs)
    section_5_7(tuning_runs)

    print("\n" + "=" * 55)
    print("  All analysis complete.")
    print(f"  Plots saved to: {PLOTS_DIR}/")
    print("=" * 55)