"""import_existing.py — однократный импорт всех существующих JSON-результатов в БД.

Маппинг file → (exp_id, block, name) — fixed table ниже.
"""

import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path("data/eval/results/final")

# (filename, exp_id, block, name)
MAPPING = [
    # n=20 RGB (legacy)
    ("metrics_triad_baseline.json",                              "legacy_baseline_n20",            "legacy", "BLIP-base + Marian + full (legacy baseline)"),
    ("metrics_triad_E-1_blip_large.json",                        "legacy_E1_n20",                  "legacy", "BLIP-large + Marian + full"),
    ("metrics_triad_E-16_drop_theme.json",                       "legacy_E16_n20",                 "legacy", "BLIP-base + Marian + full -theme -mood"),
    ("metrics_triad_E-1+16_blip_large_drop_theme.json",          "legacy_E1+16_n20",               "legacy", "BLIP-large + Marian + full -theme -mood"),
    # n=220 (full pool)
    ("metrics_triad_baseline_n220.json",                         "legacy_baseline_n220",           "legacy", "BLIP-base + Marian + full"),
    ("metrics_triad_E-1_blip_large_n220.json",                   "legacy_E1_n220",                 "legacy", "BLIP-large + Marian + full"),
    ("metrics_triad_E-16_drop_theme_n220.json",                  "legacy_E16_n220",                "legacy", "BLIP-base + Marian + full -theme -mood"),
    ("metrics_triad_E-16b_minimal_n220.json",                    "legacy_E16b_minimal_n220",       "legacy", "BLIP-base + Marian + minimal template"),
    ("metrics_triad_E-16c_caption_only_n220.json",               "legacy_E16c_caption_only_n220",  "legacy", "BLIP-base + Marian + caption_only"),
    ("metrics_triad_E-1+16_blip_large_drop_theme_n220.json",     "legacy_E1+16_n220",              "legacy", "BLIP-large + Marian + full -theme -mood"),
    ("metrics_triad_E-1+16b_blip_large_minimal_n220.json",       "legacy_E1+16b_n220",             "legacy", "BLIP-large + Marian + minimal"),
    ("metrics_triad_E-1+16c_blip_large_caption_only_n220.json",  "legacy_E1+16c_n220",             "legacy", "BLIP-large + Marian + caption_only"),
    # E-9 translator swap (legacy)
    ("metrics_triad_E-9_nllb_n220.json",                         "legacy_E9_nllb_n220",            "legacy", "BLIP-base + NLLB-200 + full"),
    ("metrics_triad_E-9c_nllb_caption_only_n220.json",           "legacy_E9c_nllb_caption_only_n220","legacy","BLIP-base + NLLB-200 + caption_only"),
    ("metrics_triad_E-1+9_blip_large_nllb_n220.json",            "legacy_E1+9_n220",               "legacy", "BLIP-large + NLLB-200 + full"),
    ("metrics_triad_E-1+9c_blip_large_nllb_caption_only_n220.json","legacy_E1+9c_n220",           "legacy", "BLIP-large + NLLB-200 + caption_only"),
    # proposed (last clean rerun)
    ("metrics_proposed_retrieval_n220.json",  "proposed_retrieval_v1_n220",  "A",  "Proposed retrieval (BLIP-large + Marian + caption_only) — pre-postproc"),
    ("metrics_proposed_display_n220.json",    "proposed_display_v1_n220",    "A",  "Proposed display (BLIP-large + Marian + full -theme -mood) — pre-postproc"),
    # new smoke-test (postproc applied)
    ("metrics_proposed_display_n20_postproc.json", "E00_postproc_n20", "A", "Proposed display + postproc — smoke test on RGB-20"),
]

# (filename, exp_id) — SDS files
SDS_MAPPING = [
    ("sds_proposed_retrieval_n220.json", "proposed_retrieval_v1_n220"),
    ("sds_proposed_display_n220.json",   "proposed_display_v1_n220"),
]


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"    [ERROR] {res.stderr}")
    else:
        print(f"    {res.stdout.strip()}")


def main():
    for fname, exp_id, block, name in MAPPING:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"[SKIP] missing: {path}")
            continue
        run([
            sys.executable, "scripts/exp_db.py", "ingest",
            "--json", str(path),
            "--exp-id", exp_id,
            "--block", block,
            "--name", name,
            "--replace",
        ])

    for fname, exp_id in SDS_MAPPING:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"[SKIP] missing: {path}")
            continue
        run([
            sys.executable, "scripts/exp_db.py", "ingest-sds",
            "--json", str(path),
            "--exp-id", exp_id,
        ])


if __name__ == "__main__":
    main()
