#!/usr/bin/env python3
"""
compute_kappa.py — Multi-source kappa comparison for the pilot hand-coding.

Compares your human scores (from sample_to_code.csv) against every LLM-judge
source we have on disk:

  * v1 single-judge (coding_sheet_scored.csv)       — the noon pass, score column = "score"
  * v2 single-judge (coding_sheet_v2_scored.csv)    — the afternoon pass, score column = "score"
  * 5-judge panel    (sample_5judge_aggregated.csv) — J1, J2, J3, J4, J5, and median
  * 4-judge consensus (computed from panel, excludes J5 outlier)

Joins on blind_id (since that's what the 5-judge file uses). For the v1/v2
coding sheets which have both blind_id and call_id, we use blind_id to match
the 30-row sample.

Outputs a single table: source, n, kappa, 95% CI, exact agreement %, within-1 %.

Pure stdlib. No numpy/scipy.

Usage:
    python3 compute_kappa.py

Author: Auric Hardcastle — SDS 2025-26
"""

import csv
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
HUMAN_CSV = SCRIPT_DIR / "sample_to_code.csv"
V1_SCORED = SCRIPT_DIR / "coding_sheet_scored.csv"
V2_SCORED = SCRIPT_DIR / "coding_sheet_v2_scored.csv"
PANEL_CSV = SCRIPT_DIR / "sample_5judge_aggregated.csv"

N_BOOTSTRAP = 2000
SEED = 20260409
K = 4  # number of ordinal categories (0..3)


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------

def weighted_kappa(pairs, k=K):
    if not pairs:
        return float("nan")
    n = len(pairs)
    obs = [[0] * k for _ in range(k)]
    for a, b in pairs:
        if 0 <= a < k and 0 <= b < k:
            obs[a][b] += 1
    row_tot = [sum(obs[i]) for i in range(k)]
    col_tot = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    num = 0.0
    den = 0.0
    for i in range(k):
        for j in range(k):
            expected = (row_tot[i] * col_tot[j]) / n if n else 0
            num += w[i][j] * obs[i][j]
            den += w[i][j] * expected
    if den == 0:
        return 1.0 if num == 0 else float("nan")
    return 1 - (num / den)


def bootstrap_ci(pairs, n_iter=N_BOOTSTRAP):
    if not pairs:
        return (float("nan"), float("nan"))
    rng = random.Random(SEED)
    estimates = []
    n = len(pairs)
    for _ in range(n_iter):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        k_est = weighted_kappa(sample)
        if k_est == k_est:  # not NaN
            estimates.append(k_est)
    if not estimates:
        return (float("nan"), float("nan"))
    estimates.sort()
    lo_idx = max(0, int(0.025 * len(estimates)))
    hi_idx = min(len(estimates) - 1, int(0.975 * len(estimates)) - 1)
    return estimates[lo_idx], estimates[hi_idx]


# ----------------------------------------------------------------------------
# Load helpers
# ----------------------------------------------------------------------------

def load_by_blind(path, score_col):
    """Return {blind_id: int score} from a CSV, skipping blanks and non-ints."""
    out = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bid = (row.get("blind_id") or "").strip()
            raw = (row.get(score_col) or "").strip()
            if not bid or not raw:
                continue
            try:
                out[bid] = int(raw)
            except ValueError:
                continue
    return out


def load_panel(path):
    """Return {blind_id: {J1..J5, median, consensus_4j}}."""
    out = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bid = (row.get("blind_id") or "").strip()
            if not bid:
                continue
            rec = {}
            for key in ("J1", "J2", "J3", "J4", "J5", "median"):
                v = (row.get(key) or "").strip()
                try:
                    rec[key] = int(float(v)) if v else None
                except ValueError:
                    rec[key] = None
            # 4-judge consensus: median of J1..J4 (excludes outlier J5)
            j14 = [rec[k] for k in ("J1", "J2", "J3", "J4") if rec[k] is not None]
            if j14:
                j14.sort()
                mid = len(j14) // 2
                rec["consensus_4j"] = j14[mid] if len(j14) % 2 == 1 else round((j14[mid - 1] + j14[mid]) / 2)
            else:
                rec["consensus_4j"] = None
            out[bid] = rec
    return out


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def report_source(name, human_scores, other_scores):
    shared = sorted(set(human_scores) & set(other_scores))
    pairs = [(human_scores[b], other_scores[b]) for b in shared if other_scores[b] is not None]
    if not pairs:
        print(f"  {name:28s}  (no overlap)")
        return
    k = weighted_kappa(pairs)
    lo, hi = bootstrap_ci(pairs)
    exact = sum(1 for a, b in pairs if a == b) / len(pairs) * 100
    within1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs) * 100
    verdict = (
        "excellent" if k >= 0.80 else
        "substantial" if k >= 0.70 else
        "moderate" if k >= 0.60 else
        "poor"
    )
    print(f"  {name:28s}  n={len(pairs):2d}  κ={k:+.3f}  "
          f"CI=[{lo:+.3f},{hi:+.3f}]  exact={exact:5.1f}%  ±1={within1:5.1f}%  [{verdict}]")


def main():
    human = load_by_blind(HUMAN_CSV, "score_0_to_3")
    if not human:
        print(f"ERROR: no human scores found in {HUMAN_CSV.name}.")
        print("Fill in the score_0_to_3 column for the 30 rows and re-run.")
        sys.exit(1)
    print(f"Loaded {len(human)} human scores from {HUMAN_CSV.name}")
    from collections import Counter
    print(f"  human score distribution: {dict(sorted(Counter(human.values()).items()))}")

    v1 = load_by_blind(V1_SCORED, "score")
    v2 = load_by_blind(V2_SCORED, "score")
    panel = load_panel(PANEL_CSV)

    print("\nComparisons vs. human scores (Cohen's weighted kappa, quadratic weights):")
    print("-" * 92)

    if v1:
        report_source("v1 single-judge (noon)", human, v1)
    if v2:
        report_source("v2 single-judge (afternoon)", human, v2)

    if panel:
        for judge_key in ("J1", "J2", "J3", "J4", "J5"):
            j_scores = {b: rec[judge_key] for b, rec in panel.items() if rec.get(judge_key) is not None}
            if j_scores:
                report_source(f"5-panel: {judge_key}", human, j_scores)
        median_scores = {b: rec["median"] for b, rec in panel.items() if rec.get("median") is not None}
        if median_scores:
            report_source("5-panel: median (all 5)", human, median_scores)
        consensus_scores = {b: rec["consensus_4j"] for b, rec in panel.items() if rec.get("consensus_4j") is not None}
        if consensus_scores:
            report_source("4-panel consensus (no J5)", human, consensus_scores)

    print("-" * 92)
    print("\nInterpretation guide:")
    print("  κ ≥ 0.80  excellent — judge is a defensible auxiliary coder")
    print("  κ ≥ 0.70  substantial — judge viable with reported limitations")
    print("  κ ≥ 0.60  moderate   — borderline, discuss")
    print("  κ <  0.60 poor       — not viable, full human coding needed")
    print("\nFor the paper: report the judge with the highest κ as your auxiliary method,")
    print("with the CI, exact agreement %, and the confusion matrix in the appendix.")


if __name__ == "__main__":
    main()
