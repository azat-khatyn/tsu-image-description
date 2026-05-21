#!/usr/bin/env bash
# Run all 4 Block A configs on semantic_testset (n=14), compute SDS, ingest to DB.
set -e

REFS=data/eval/semantic_testset.jsonl
OUT_DIR=data/eval/results/final

run_and_log() {
    local exp_id="$1"
    local out_json="$2"
    local sds_json="$3"
    local name="$4"
    shift 4
    local cfg_args=("$@")

    echo
    echo "=================================================================="
    echo ">> $exp_id"
    echo "=================================================================="

    PYTHONPATH=src python scripts/evaluate.py \
        --caption-backend blip1 \
        --references "$REFS" \
        --output "$out_json" \
        "${cfg_args[@]}" 2>&1 | tail -3

    python scripts/compute_sds.py \
        --input "$out_json" \
        --output "$sds_json" \
        --mode keyword 2>&1 | tail -3

    python scripts/exp_db.py ingest \
        --json "$out_json" \
        --exp-id "$exp_id" \
        --block A \
        --name "$name" \
        --replace
    python scripts/exp_db.py ingest-sds \
        --json "$sds_json" \
        --exp-id "$exp_id"
}

# E00: BLIP-large + Marian + display (full -theme -mood)
run_and_log \
    "E00_semtest" \
    "$OUT_DIR/metrics_E00_semtest.json" \
    "$OUT_DIR/sds_E00_semtest.json" \
    "E00 BLIP-large + Marian + postproc + full -theme -mood (semantic testset)" \
    --finetuned Salesforce/blip-image-captioning-large \
    --template-mode full --drop-theme --drop-mood

# E01: BLIP-base + Marian + display
run_and_log \
    "E01_semtest" \
    "$OUT_DIR/metrics_E01_semtest.json" \
    "$OUT_DIR/sds_E01_semtest.json" \
    "E01 BLIP-base + Marian + postproc + full -theme -mood (semantic testset)" \
    --finetuned Salesforce/blip-image-captioning-base \
    --template-mode full --drop-theme --drop-mood

# E05a: BLIP-large + caption_only (retrieval)
run_and_log \
    "E05a_semtest" \
    "$OUT_DIR/metrics_E05a_semtest.json" \
    "$OUT_DIR/sds_E05a_semtest.json" \
    "E05a BLIP-large + Marian + postproc + caption_only (semantic testset)" \
    --finetuned Salesforce/blip-image-captioning-large \
    --template-mode caption_only

# E05c: BLIP-large + full + theme + mood
run_and_log \
    "E05c_semtest" \
    "$OUT_DIR/metrics_E05c_semtest.json" \
    "$OUT_DIR/sds_E05c_semtest.json" \
    "E05c BLIP-large + Marian + postproc + full +theme +mood (semantic testset)" \
    --finetuned Salesforce/blip-image-captioning-large \
    --template-mode full

echo
echo "=================================================================="
echo "ALL DONE"
echo "=================================================================="
python scripts/exp_db.py list | tail -10
