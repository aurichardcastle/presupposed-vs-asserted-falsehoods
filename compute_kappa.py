#!/usr/bin/env python3
"""
compute_kappa.py — Judge agreement statistics.

Computes inter-rater agreement between the primary LLM judge (Claude Sonnet 4.6)
and the cross-validation judge (GPT-4o) on the stratified 199-response sample
stored in cross_validated_scores.csv.

Reports:
  - Exact agreement %
  - Within-one-point agreement %
  - Cohen's weighted kappa (quadratic weights) with bootstrap 95% CI
  - Confusion matrix
"""

import csv
import random
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / "cross_validated_scores.csv"
N_BOOTSTRAP = 2000
SEED = 42
K = 4  # ordinal categories 0..3


def load_pairs(path):
    """Return [(primary_score, gpt4o_score), ...] from cross_validated_scores.csv."""
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                a = int(row["original_score"])
                b = int(row["gpt4o_score"])
            except (ValueError, KeyError):
                continue
            if b == -1 or a == -1:
                continue
            pairs.append((a, b))
    return pairs


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
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            expected = (row_tot[i] * col_tot[j]) / n if n else 0
            num += w[i][j] * obs[i][j]
            den += w[i][j] * expected
    if den == 0:
        return 1.0 if num == 0 else float("nan")
    return 1 - (num / den)


def bootstrap_ci(pairs, n_iter=N_BOOTSTRAP):
    rng = random.Random(SEED)
    estimates = []
    n = len(pairs)
    for _ in range(n_iter):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        k_est = weighted_kappa(sample)
        if k_est == k_est:
            estimates.append(k_est)
    estimates.sort()
    lo = estimates[max(0, int(0.025 * len(estimates)))]
    hi = estimates[min(len(estimates) - 1, int(0.975 * len(estimates)) - 1)]
    return lo, hi


def confusion_matrix(pairs, k=K):
    mat = [[0] * k for _ in range(k)]
    for a, b in pairs:
        if 0 <= a < k and 0 <= b < k:
            mat[a][b] += 1
    return mat


def main():
    pairs = load_pairs(INPUT_CSV)
    n = len(pairs)
    if n == 0:
        print(f"ERROR: no valid pairs found in {INPUT_CSV.name}.")
        return

    exact = sum(1 for a, b in pairs if a == b)
    within1 = sum(1 for a, b in pairs if abs(a - b) <= 1)
    k = weighted_kappa(pairs)
    lo, hi = bootstrap_ci(pairs)

    print(f"Judge agreement: Claude Sonnet 4.6 (primary) vs GPT-4o (cross-validation)")
    print(f"Source: {INPUT_CSV.name}")
    print(f"N pairs: {n}")
    print()
    print(f"Exact agreement:     {exact}/{n} = {100*exact/n:.1f}%")
    print(f"Within-1 agreement:  {within1}/{n} = {100*within1/n:.1f}%")
    print(f"Weighted kappa:      {k:+.3f}  (95% CI [{lo:+.3f}, {hi:+.3f}])")
    print()
    print("Confusion matrix (rows = primary, cols = GPT-4o):")
    mat = confusion_matrix(pairs)
    print(f"          GPT-4o=0   =1   =2   =3")
    for i, row in enumerate(mat):
        print(f"  primary={i}  " + "  ".join(f"{v:4d}" for v in row))
    print()
    print("Score distributions:")
    print(f"  primary: {dict(sorted(Counter(a for a, _ in pairs).items()))}")
    print(f"  gpt-4o:  {dict(sorted(Counter(b for _, b in pairs).items()))}")


if __name__ == "__main__":
    main()
