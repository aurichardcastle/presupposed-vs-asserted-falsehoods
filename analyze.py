#!/usr/bin/env python3
"""
Statistical analysis for sycophancy research.
Reads scored responses, computes descriptive statistics, runs Friedman tests
with pairwise Wilcoxon signed-rank post-hoc tests (Bonferroni corrected),
generates APA-formatted tables and visualizations.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

OUTPUT_DIR = "results"

MODELS = ["gpt-4o", "claude-sonnet-4-5", "gemini-2.0-flash"]
MODEL_LABELS = {
    "gpt-4o": "GPT-4o",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
}
CONDITIONS = ["A", "B", "C", "D"]
CONDITION_LABELS = {
    "A": "Presupposed-False",
    "B": "Asserted-False",
    "C": "Presupposed-True (Control)",
    "D": "Authority + Presupposed-False",
}


def load_scored(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["score"].apply(lambda x: str(x).isdigit())]
    df["score"] = df["score"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["run"] = df["run"].astype(int)
    return df


def compute_median_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute median score across 3 runs for each item-model-condition."""
    grouped = (
        df.groupby(["item_id", "model", "condition"])["score"]
        .median()
        .reset_index()
    )
    grouped.columns = ["item_id", "model", "condition", "median_score"]
    return grouped


def descriptive_stats(medians: pd.DataFrame) -> pd.DataFrame:
    """Compute M, SD, N, Mdn, Min, Max per model × condition."""
    rows = []
    for model in MODELS:
        for cond in CONDITIONS:
            subset = medians[
                (medians["model"] == model) & (medians["condition"] == cond)
            ]["median_score"]
            rows.append({
                "Model": MODEL_LABELS[model],
                "Condition": CONDITION_LABELS[cond],
                "N": len(subset),
                "M": round(subset.mean(), 3),
                "SD": round(subset.std(), 3),
                "Mdn": round(subset.median(), 1),
                "Min": subset.min(),
                "Max": subset.max(),
            })
    return pd.DataFrame(rows)


def run_friedman_tests(medians: pd.DataFrame) -> list[dict]:
    """Run Friedman test per model comparing all available conditions."""
    active_conds = sorted(medians["condition"].unique())
    results = []
    for model in MODELS:
        model_data = medians[medians["model"] == model]
        samples = []
        conds_used = []
        for cond in active_conds:
            cond_data = model_data[model_data["condition"] == cond].sort_values("item_id")
            if len(cond_data) > 0:
                samples.append(cond_data["median_score"].values)
                conds_used.append(cond)

        n = min(len(s) for s in samples)
        samples = [s[:n] for s in samples]
        df_val = len(samples) - 1

        stat, p = stats.friedmanchisquare(*samples)
        results.append({
            "Model": MODEL_LABELS[model],
            "N": n,
            "Chi-square": round(stat, 3),
            "df": df_val,
            "p": p,
            "p_formatted": f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}",
            "significant": p < 0.05,
            "samples": samples,
            "conditions_used": conds_used,
        })
    return results


def run_posthoc_wilcoxon(friedman_results: list[dict]) -> list[dict]:
    """Pairwise Wilcoxon signed-rank tests with Bonferroni correction."""
    all_pair_labels = {
        ("A", "B"): "Presupposed-False vs Asserted-False",
        ("A", "C"): "Presupposed-False vs True Control",
        ("B", "C"): "Asserted-False vs True Control",
        ("A", "D"): "Presupposed-False vs Authority+Presupposed",
        ("B", "D"): "Asserted-False vs Authority+Presupposed",
        ("C", "D"): "True Control vs Authority+Presupposed",
    }
    posthoc = []
    for fr in friedman_results:
        if not fr["significant"]:
            posthoc.append({
                "Model": fr["Model"],
                "note": "Friedman not significant; post-hoc tests not performed",
            })
            continue

        conds_used = fr.get("conditions_used", CONDITIONS)
        samples_dict = dict(zip(conds_used, fr["samples"]))
        pairs = [(c1, c2) for i, c1 in enumerate(conds_used) for c2 in conds_used[i+1:]]
        pair_labels = all_pair_labels
        n_comparisons = len(pairs)
        for c1, c2 in pairs:
            s1 = samples_dict[c1]
            s2 = samples_dict[c2]
            diff = s1 - s2
            if np.all(diff == 0):
                posthoc.append({
                    "Model": fr["Model"],
                    "Comparison": pair_labels[(c1, c2)],
                    "W": "N/A",
                    "p_raw": 1.0,
                    "p_bonferroni": 1.0,
                    "p_formatted": "1.0000",
                    "significant": False,
                    "note": "All differences are zero",
                })
                continue

            try:
                w_stat, p_raw = stats.wilcoxon(s1, s2, alternative="two-sided")
            except ValueError:
                posthoc.append({
                    "Model": fr["Model"],
                    "Comparison": pair_labels[(c1, c2)],
                    "W": "N/A",
                    "p_raw": 1.0,
                    "p_bonferroni": 1.0,
                    "p_formatted": "N/A",
                    "significant": False,
                    "note": "Could not compute (identical distributions)",
                })
                continue

            p_bonf = min(p_raw * n_comparisons, 1.0)
            posthoc.append({
                "Model": fr["Model"],
                "Comparison": pair_labels[(c1, c2)],
                "W": round(w_stat, 3),
                "p_raw": round(p_raw, 6),
                "p_bonferroni": round(p_bonf, 6),
                "p_formatted": f"{p_bonf:.4f}" if p_bonf >= 0.0001 else f"{p_bonf:.2e}",
                "significant": p_bonf < 0.05,
            })
    return posthoc


def compute_score_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many responses fall in each score category per model × condition."""
    rows = []
    active_conds = [c for c in ["A", "B", "D"] if c in df["condition"].values]
    for model in MODELS:
        for cond in active_conds:
            subset = df[(df["model"] == model) & (df["condition"] == cond)]
            total = len(subset)
            for score_val in [0, 1, 2, 3]:
                count = len(subset[subset["score"] == score_val])
                pct = round(100 * count / total, 1) if total > 0 else 0
                rows.append({
                    "Model": MODEL_LABELS[model],
                    "Condition": CONDITION_LABELS[cond],
                    "Score": score_val,
                    "Count": count,
                    "Percentage": pct,
                })
    return pd.DataFrame(rows)


def plot_mean_scores(desc_stats: pd.DataFrame, output_dir: str):
    """Figure 1: Bar chart of mean sycophancy scores by model and condition."""
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_data = desc_stats[desc_stats["Condition"] != "Presupposed-True (Control)"].copy()
    plot_conds = [c for c in ["Presupposed-False", "Asserted-False", "Authority + Presupposed-False"]
                  if c in plot_data["Condition"].values]

    models_list = [MODEL_LABELS[m] for m in MODELS]
    x = np.arange(len(models_list))
    n_conds = len(plot_conds)
    width = 0.8 / n_conds

    colors = {
        "Presupposed-False": "#5B8FB9",
        "Asserted-False": "#B6533A",
        "Authority + Presupposed-False": "#6B3FA0",
    }

    for i, cond in enumerate(plot_conds):
        cond_data = plot_data[plot_data["Condition"] == cond]
        means = [cond_data[cond_data["Model"] == m]["M"].values[0] for m in models_list]
        sds = [cond_data[cond_data["Model"] == m]["SD"].values[0] for m in models_list]
        offset = (i - (n_conds - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=sds,
            label=cond, color=colors[cond], capsize=4,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Model", fontsize=12, labelpad=10)
    ax.set_ylabel("Mean Sycophancy Score (0–3)", fontsize=12, labelpad=10)
    ax.set_title(
        "Figure 1. Mean Sycophancy Scores by Model and Condition",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, fontsize=11)
    ax.set_ylim(0, 3.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.legend(title="Condition", fontsize=10, title_fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_mean_scores.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_score_distributions(dist_df: pd.DataFrame, output_dir: str):
    """Figure 2: Stacked bar chart showing score distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    score_colors = {0: "#2E7D32", 1: "#FBC02D", 2: "#E65100", 3: "#B71C1C"}
    score_labels = {
        0: "Explicit Correction",
        1: "Hedged Correction",
        2: "Silent Acceptance",
        3: "Elaborative Acceptance",
    }

    has_d = "Authority + Presupposed-False" in dist_df["Condition"].values
    conds = ["Presupposed-False", "Asserted-False"]
    cond_xlabels = ["Presupposed\nFalse", "Asserted\nFalse"]
    if has_d:
        conds.append("Authority + Presupposed-False")
        cond_xlabels.append("Authority +\nPresupposed")

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        model_label = MODEL_LABELS[model]
        model_data = dist_df[dist_df["Model"] == model_label]

        x = np.arange(len(conds))
        bottom = np.zeros(len(conds))
        for score_val in [0, 1, 2, 3]:
            heights = []
            for cond in conds:
                row = model_data[
                    (model_data["Condition"] == cond)
                    & (model_data["Score"] == score_val)
                ]
                heights.append(row["Percentage"].values[0] if len(row) > 0 else 0)
            ax.bar(
                x, heights, 0.5, bottom=bottom,
                color=score_colors[score_val],
                label=score_labels[score_val] if idx == 0 else "",
                edgecolor="white", linewidth=0.5,
            )
            bottom += heights

        ax.set_title(model_label, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_xlabels, fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percentage (%)" if idx == 0 else "", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(
        loc="upper left", fontsize=8,
        title="Score Category", title_fontsize=9, bbox_to_anchor=(0, 1),
    )
    fig.suptitle(
        "Figure 2. Score Category Distributions by Model and Condition",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_item_heatmap(medians: pd.DataFrame, output_dir: str):
    """Figure 3: Heatmap of per-item median scores (A vs B) across models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        model_data = medians[
            (medians["model"] == model) & (medians["condition"].isin(["A", "B"]))
        ]
        pivot = model_data.pivot(
            index="item_id", columns="condition", values="median_score"
        )
        pivot.columns = ["Presupposed", "Asserted"]
        pivot = pivot.sort_index()

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=3,
            annot=True,
            fmt=".0f",
            cbar=idx == 2,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")
        ax.set_ylabel("Item ID" if idx == 0 else "", fontsize=11)
        ax.set_xlabel("")

    fig.suptitle(
        "Figure 3. Per-Item Median Sycophancy Scores by Condition and Model",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_results_text(
    desc_stats: pd.DataFrame,
    friedman_results: list[dict],
    posthoc_results: list[dict],
    dist_df: pd.DataFrame,
    exclusions: dict,
    output_dir: str,
):
    """Generate APA-style results section text."""
    lines = []
    lines.append("RESULTS\n")
    lines.append("=" * 70)

    # Exclusions
    total_responses = exclusions.pop("total_collected", len(df))
    lines.append(f"\nData Exclusions")
    lines.append(f"Of the {total_responses} total responses collected:")
    lines.append(f"  Truncated: {exclusions.get('truncated', 0)}")
    lines.append(f"  Empty: {exclusions.get('empty', 0)}")
    lines.append(f"  Refusals: {exclusions.get('refusal', 0)}")
    lines.append(f"  Errors: {exclusions.get('error', 0)}")
    total_excluded = sum(exclusions.values())
    lines.append(f"  Total excluded: {total_excluded}")
    lines.append(f"  Responses analyzed: {total_responses - total_excluded}")

    # Condition C validation
    lines.append(f"\nCondition C (True-Presupposition Control) Validation")
    lines.append(f"All Condition C responses were scored 0 by default, as the rubric")
    lines.append(f"measures acceptance of false content and no false content was present.")

    # Descriptive stats table
    lines.append(f"\n{'='*70}")
    lines.append(f"Table 1. Descriptive Statistics of Sycophancy Scores by Model and Condition")
    lines.append(f"{'='*70}")
    lines.append(
        f"{'Model':<25} {'Condition':<30} {'N':>4} {'M':>7} {'SD':>7} {'Mdn':>5}"
    )
    lines.append(f"{'-'*70}")
    for _, row in desc_stats.iterrows():
        lines.append(
            f"{row['Model']:<25} {row['Condition']:<30} {row['N']:>4} "
            f"{row['M']:>7.3f} {row['SD']:>7.3f} {row['Mdn']:>5.1f}"
        )
    lines.append(f"{'='*70}")
    lines.append(f"Note. Scores range from 0 (explicit correction) to 3 (elaborative")
    lines.append(f"acceptance). Each cell reports the median score across 3 runs per")
    lines.append(f"item, aggregated across 40 items. M = mean, SD = standard deviation,")
    lines.append(f"Mdn = median, N = number of items.\n")

    # Friedman test results
    lines.append(f"{'='*70}")
    lines.append(f"Table 2. Friedman Test Results by Model")
    lines.append(f"{'='*70}")
    lines.append(
        f"{'Model':<25} {'N':>4} {'χ²(2)':>10} {'p':>12} {'Sig.':>6}"
    )
    lines.append(f"{'-'*70}")
    for fr in friedman_results:
        sig = "Yes" if fr["significant"] else "No"
        lines.append(
            f"{fr['Model']:<25} {fr['N']:>4} {fr['Chi-square']:>10.3f} "
            f"{fr['p_formatted']:>12} {sig:>6}"
        )
    lines.append(f"{'='*70}")
    lines.append(f"Note. The Friedman test is a non-parametric alternative to the")
    lines.append(f"repeated-measures ANOVA for ordinal data across related conditions")
    lines.append(f"(A: presupposed-false, B: asserted-false, C: true control,")
    lines.append(f"D: authority + presupposed-false).\n")

    # Post-hoc results
    lines.append(f"{'='*70}")
    lines.append(f"Table 3. Pairwise Wilcoxon Signed-Rank Post-Hoc Tests (Bonferroni Corrected)")
    lines.append(f"{'='*70}")
    lines.append(
        f"{'Model':<22} {'Comparison':<38} {'W':>8} {'p (corr.)':>12} {'Sig.':>6}"
    )
    lines.append(f"{'-'*70}")
    for ph in posthoc_results:
        if "note" in ph and "not significant" in ph.get("note", ""):
            lines.append(f"{ph['Model']:<22} {ph['note']}")
            continue
        sig = "Yes" if ph.get("significant", False) else "No"
        w_str = str(ph.get("W", "N/A"))
        lines.append(
            f"{ph['Model']:<22} {ph.get('Comparison',''):<38} {w_str:>8} "
            f"{ph.get('p_formatted',''):>12} {sig:>6}"
        )
    lines.append(f"{'='*70}")
    lines.append(f"Note. Bonferroni correction applied for 3 pairwise comparisons")
    lines.append(f"(adjusted α = .0167). W = Wilcoxon signed-rank statistic.\n")

    # Score distribution summary
    active_dist_conds = [c for c in ["Presupposed-False", "Asserted-False", "Authority + Presupposed-False"]
                         if c in dist_df["Condition"].values]
    lines.append(f"{'='*80}")
    lines.append(f"Table 4. Score Category Distributions (False-Premise Conditions)")
    lines.append(f"{'='*80}")
    lines.append(
        f"{'Model':<22} {'Condition':<30} {'Score 0':>10} {'Score 1':>10} "
        f"{'Score 2':>10} {'Score 3':>10}"
    )
    lines.append(f"{'-'*80}")
    for model in MODELS:
        model_label = MODEL_LABELS[model]
        for cond in active_dist_conds:
            cond_data = dist_df[
                (dist_df["Model"] == model_label) & (dist_df["Condition"] == cond)
            ]
            scores_str = []
            for sv in [0, 1, 2, 3]:
                row = cond_data[cond_data["Score"] == sv]
                if len(row) > 0:
                    scores_str.append(f"{row['Count'].values[0]} ({row['Percentage'].values[0]}%)")
                else:
                    scores_str.append("0 (0.0%)")
            lines.append(
                f"{model_label:<22} {cond:<30} "
                + "".join(f"{s:>14}" for s in scores_str)
            )
    lines.append(f"{'='*80}")
    lines.append(f"Note. Percentages are of total responses per model-condition cell.\n")

    # Narrative results per model
    lines.append(f"\n{'='*70}")
    lines.append(f"NARRATIVE RESULTS BY MODEL")
    lines.append(f"{'='*70}\n")

    for fr in friedman_results:
        model = fr["Model"]
        lines.append(f"\n{model}")
        lines.append(f"{'-' * len(model)}")

        def get_stats(cond_label):
            s = desc_stats[(desc_stats["Model"] == model) & (desc_stats["Condition"] == cond_label)]
            return s.iloc[0] if len(s) > 0 else None

        stats_a = get_stats("Presupposed-False")
        stats_b = get_stats("Asserted-False")
        stats_c = get_stats("Presupposed-True (Control)")
        stats_d = get_stats("Authority + Presupposed-False")

        n_conds = len(fr.get("conditions_used", CONDITIONS))
        lines.append(
            f"A Friedman test was conducted to compare sycophancy scores across "
            f"{n_conds} conditions for {model}. The test revealed a "
            f"{'statistically significant' if fr['significant'] else 'non-significant'} "
            f"difference across conditions, χ²({fr['df']}, N = {fr['N']}) = {fr['Chi-square']:.3f}, "
            f"p = {fr['p_formatted']}."
        )

        if stats_a is not None and stats_b is not None:
            lines.append(
                f"\nPresupposed-false prompts (Condition A) yielded a mean sycophancy "
                f"score of M = {stats_a['M']:.3f} (SD = {stats_a['SD']:.3f}, "
                f"N = {int(stats_a['N'])}), while asserted-false prompts (Condition B) "
                f"yielded M = {stats_b['M']:.3f} (SD = {stats_b['SD']:.3f}, "
                f"N = {int(stats_b['N'])})."
            )

        if stats_c is not None:
            lines.append(
                f"The true-presupposition control (Condition C) yielded "
                f"M = {stats_c['M']:.3f} (SD = {stats_c['SD']:.3f}, N = {int(stats_c['N'])})."
            )

        if stats_d is not None:
            lines.append(
                f"Authority-framed presupposed-false prompts (Condition D) yielded "
                f"M = {stats_d['M']:.3f} (SD = {stats_d['SD']:.3f}, N = {int(stats_d['N'])}), "
                f"{'higher' if stats_d['M'] > stats_a['M'] else 'comparable to or lower'} "
                f"than the unframed presupposed condition."
            )

        model_posthoc = [
            p for p in posthoc_results
            if p.get("Model") == model and "Comparison" in p
        ]
        if model_posthoc:
            lines.append("\nPost-hoc pairwise comparisons (Wilcoxon signed-rank, Bonferroni corrected):")
            for ph in model_posthoc:
                sig_text = "significant" if ph["significant"] else "not significant"
                lines.append(
                    f"  {ph['Comparison']}: W = {ph['W']}, "
                    f"p = {ph['p_formatted']} ({sig_text})"
                )
        lines.append("")

    text = "\n".join(lines)

    with open(f"{output_dir}/results.txt", "w") as f:
        f.write(text)

    return text


def main():
    script_dir = Path(__file__).parent
    input_path = script_dir / INPUT_CSV
    output_dir = script_dir / OUTPUT_DIR

    # Auto-detect which dataset to use
    merged_path = script_dir / "scored_responses_all.csv"
    base_path = script_dir / "scored_responses.csv"
    if merged_path.exists():
        input_path = merged_path
        print(f"Using merged dataset: {merged_path.name}")
    elif base_path.exists():
        input_path = base_path
        print(f"Using base dataset: {base_path.name}")
    else:
        print("Error: No scored responses found. Run autoscore.py first.")
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)

    print("Loading scored responses...")
    df = load_scored(str(input_path))
    print(f"  Loaded {len(df)} scored responses")

    active_conditions = sorted(df["condition"].unique())
    print(f"  Conditions found: {active_conditions}")

    # Count exclusions from original responses
    exclusions = {"truncated": 0, "empty": 0, "refusal": 0, "error": 0, "total_collected": 0}
    for resp_file in ["responses.csv", "responses_d.csv"]:
        orig_path = script_dir / resp_file
        if orig_path.exists():
            orig = pd.read_csv(str(orig_path))
            exclusions["total_collected"] += len(orig)
            exclusions["truncated"] += int((orig.get("is_truncated", pd.Series()) == "True").sum())
            exclusions["empty"] += int((orig.get("is_empty", pd.Series()) == "True").sum())
            exclusions["refusal"] += int((orig.get("is_refusal", pd.Series()) == "True").sum())
            exclusions["error"] += int(orig.get("error", pd.Series()).apply(lambda x: bool(x) and str(x) != "").sum())

    print("Computing median scores (across 3 runs per item)...")
    medians = compute_median_scores(df)

    print("Computing descriptive statistics...")
    desc = descriptive_stats(medians)

    print("Running Friedman tests...")
    friedman = run_friedman_tests(medians)
    for fr in friedman:
        sig = "***" if fr["significant"] else ""
        print(f"  {fr['Model']}: χ²(2) = {fr['Chi-square']:.3f}, p = {fr['p_formatted']} {sig}")

    print("Running post-hoc Wilcoxon signed-rank tests...")
    posthoc = run_posthoc_wilcoxon(friedman)

    print("Computing score distributions...")
    dist = compute_score_distributions(df)

    print("Generating visualizations...")
    plot_mean_scores(desc, str(output_dir))
    print("  Saved figure1_mean_scores.png")

    plot_score_distributions(dist, str(output_dir))
    print("  Saved figure2_distributions.png")

    plot_item_heatmap(medians, str(output_dir))
    print("  Saved figure3_heatmap.png")

    print("Generating results text...")
    results_text = generate_results_text(desc, friedman, posthoc, dist, exclusions, str(output_dir))

    # Save descriptive stats as CSV too
    desc.to_csv(f"{output_dir}/descriptive_stats.csv", index=False)
    pd.DataFrame(friedman).drop(columns=["samples"]).to_csv(
        f"{output_dir}/friedman_results.csv", index=False
    )
    pd.DataFrame(posthoc).to_csv(f"{output_dir}/posthoc_results.csv", index=False)
    dist.to_csv(f"{output_dir}/score_distributions.csv", index=False)

    print(f"\n{'='*70}")
    print(results_text)
    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
