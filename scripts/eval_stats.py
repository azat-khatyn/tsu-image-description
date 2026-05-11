"""eval_stats.py — bootstrap CI и paired-test для триады (I-4).

Использование как библиотеки:

    from scripts.eval_stats import bootstrap_ci, paired_bootstrap

    ci_low, ci_high = bootstrap_ci(values, n_iter=1000, conf=0.95)
    p = paired_bootstrap(a_values, b_values, n_iter=1000)

Использование как CLI на готовых per-item метриках:

    python scripts/eval_stats.py \
        --baseline data/eval/results/metrics_triad_v1.json \
        --candidate data/eval/results/metrics_triad_v2.json \
        --metric CLIPScore_RU_archive_ru
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------
def bootstrap_ci(values, n_iter=1000, conf=0.95, seed=42):
    """Percentile bootstrap CI для среднего."""
    if not values:
        return None, None

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None, None

    rng = np.random.default_rng(seed)
    means = np.empty(n_iter)
    n = arr.size
    for i in range(n_iter):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = sample.mean()

    alpha = (1 - conf) / 2
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1 - alpha))
    return low, high


def paired_bootstrap(a, b, n_iter=1000, seed=42):
    """Paired bootstrap: P(mean(a-b) <= 0) — двусторонний test через симметрию.

    Возвращает p-value для гипотезы H0: mean(a) == mean(b).
    Малое p — отвергаем H0 в пользу того, что распределения различны.
    """
    if len(a) != len(b):
        raise ValueError(f"Paired bootstrap requires equal-length arrays: {len(a)} vs {len(b)}")

    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    diff = arr_a - arr_b

    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    diff = diff[mask]
    if diff.size == 0:
        return None

    rng = np.random.default_rng(seed)
    n = diff.size
    obs_mean = float(diff.mean())
    boot_means = np.empty(n_iter)
    for i in range(n_iter):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means[i] = sample.mean()

    centered = boot_means - obs_mean
    p = float(np.mean(np.abs(centered) >= abs(obs_mean)))
    return p


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _extract_metric(payload, key):
    items = payload.get("per_item") or []
    return [
        it["scores"].get(key)
        for it in items
        if it.get("scores", {}).get(key) is not None
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="metrics JSON for baseline")
    parser.add_argument("--candidate", required=True, help="metrics JSON for candidate")
    parser.add_argument(
        "--metric",
        default="CLIPScore_RU_archive_ru",
        help="metric key inside per_item.scores",
    )
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--conf", type=float, default=0.95)
    args = parser.parse_args()

    with open(args.baseline) as f:
        base = json.load(f)
    with open(args.candidate) as f:
        cand = json.load(f)

    base_vals = _extract_metric(base, args.metric)
    cand_vals = _extract_metric(cand, args.metric)

    print(f"[INFO] metric: {args.metric}")
    print(f"[INFO] baseline n={len(base_vals)}  candidate n={len(cand_vals)}")

    base_lo, base_hi = bootstrap_ci(base_vals, n_iter=args.n_iter, conf=args.conf)
    cand_lo, cand_hi = bootstrap_ci(cand_vals, n_iter=args.n_iter, conf=args.conf)

    print(
        f"\nBaseline mean={np.mean(base_vals):.4f}  "
        f"CI{int(args.conf * 100)}=[{base_lo:.4f}, {base_hi:.4f}]"
    )
    print(
        f"Candidate mean={np.mean(cand_vals):.4f}  "
        f"CI{int(args.conf * 100)}=[{cand_lo:.4f}, {cand_hi:.4f}]"
    )

    # Paired test only valid if same eval pool size
    if len(base_vals) == len(cand_vals):
        p = paired_bootstrap(cand_vals, base_vals, n_iter=args.n_iter)
        delta = float(np.mean(cand_vals) - np.mean(base_vals))
        print(f"\nDelta (cand − base): {delta:+.4f}")
        print(f"Paired bootstrap p-value: {p:.4f}")
        verdict = "SIGNIFICANT" if p < 0.1 else "not significant"
        print(f"Verdict (α=0.1): {verdict}")
    else:
        print("\n[WARN] sample sizes differ — paired test skipped")


if __name__ == "__main__":
    main()
