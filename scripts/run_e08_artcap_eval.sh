#!/usr/bin/env bash
# Evaluate all 3 ArtCap-LoRA epochs on semantic_testset (n=14) with E12_v2 pipeline.
set -e

REFS=data/eval/semantic_testset.jsonl
OUT_DIR=data/eval/results/final

for epoch in 0 1 2; do
    exp_id="E08_artcap_epoch${epoch}_semtest"
    metrics_json="$OUT_DIR/metrics_${exp_id}.json"
    sds_json="$OUT_DIR/sds_${exp_id}.json"
    checkpoint="models/blip_artcap_lora/v1_epoch_${epoch}"

    echo
    echo "=================================================================="
    echo ">> $exp_id  (checkpoint: $checkpoint)"
    echo "=================================================================="

    PYTHONPATH=src python scripts/evaluate.py \
        --caption-backend blip1 \
        --finetuned "$checkpoint" \
        --template-mode full --drop-theme --drop-mood --drop-generic-style \
        --use-llm-rewriter \
        --llm-model Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24 \
        --references "$REFS" \
        --output "$metrics_json" 2>&1 | tail -3

    python scripts/compute_sds.py \
        --input "$metrics_json" --output "$sds_json" --mode keyword 2>&1 | tail -3

    python scripts/exp_db.py ingest \
        --json "$metrics_json" \
        --exp-id "$exp_id" \
        --block B \
        --name "E08 BLIP-large + ArtCap-LoRA epoch ${epoch} + Marian + Vikhr-v2 (semantic testset)" \
        --replace
    python scripts/exp_db.py ingest-sds --json "$sds_json" --exp-id "$exp_id"
done

echo
echo "=================================================================="
echo "ALL E08 EPOCHS DONE"
echo "=================================================================="
python scripts/exp_db.py compare --exp-a E12_v2_semtest --exp-b E08_artcap_epoch0_semtest 2>&1 | head -30
echo
python scripts/exp_db.py compare --exp-a E12_v2_semtest --exp-b E08_artcap_epoch2_semtest 2>&1 | head -30
