#!/bin/bash
# Поэпизодная оценка дообученных BLIP-LoRA чекпойнтов на полном пуле-656.
# Для каждого epoch_N: CLIPScore (по источникам) + retrieval R@k в едином пуле.
# Идемпотентно: готовый output пропускается; внутри evaluate.py — свой чекпойнт.
set -u
cd "$(dirname "$0")/.."

CKPT_DIR=models/blip_semart_lora_valrun
REFS=data/eval/testset.jsonl
OUT_DIR=data/eval/results

for N in 0 1 2 3 4; do
  OUT="$OUT_DIR/metrics_lora_epoch_$N.json"
  if [ -f "$OUT" ]; then
    echo "=== epoch $N: $OUT уже есть, пропускаю ==="
    continue
  fi
  echo "=== epoch $N: оценка $CKPT_DIR/semart_val_epoch_$N ==="
  PYTHONPATH=src python scripts/evaluate.py \
    --finetuned "$CKPT_DIR/semart_val_epoch_$N" \
    --references "$REFS" \
    --output "$OUT" || { echo "epoch $N FAILED"; exit 1; }
done
echo "ALL EPOCHS DONE"
