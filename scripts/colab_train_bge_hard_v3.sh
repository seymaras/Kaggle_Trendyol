#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/content/Kaggle_Trendyol}"
EXPERIMENT_ID="${EXPERIMENT_ID:-bge_hard_v3_full}"
MODEL_NAME="${MODEL_NAME:-BAAI/bge-reranker-v2-m3}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${ROOT}/artifacts}"

cd "${ROOT}"

TRAIN_FILE="${TRAIN_FILE:-}"
if [[ -z "${TRAIN_FILE}" ]]; then
  for candidate in \
    "${ROOT}/train_query_negatives.parquet" \
    "${ROOT}/artifacts/experiments/query_mining_v2/train_query_negatives.parquet"; do
    if [[ -f "${candidate}" ]]; then
      TRAIN_FILE="${candidate}"
      break
    fi
  done
fi

if [[ -z "${TRAIN_FILE}" || ! -f "${TRAIN_FILE}" ]]; then
  echo "HATA: train_query_negatives.parquet bulunamadı."
  exit 2
fi

for required in items.csv terms.csv training_pairs.csv; do
  if [[ ! -f "${ROOT}/data/${required}" ]]; then
    echo "HATA: ${ROOT}/data/${required} bulunamadı."
    exit 2
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "HATA: NVIDIA GPU görünmüyor; Colab GPU runtime seçilmeli."
  exit 2
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -m pip install -q \
  "transformers>=4.45,<5" \
  "accelerate>=0.34,<2" \
  "sentencepiece>=0.2,<1" \
  "pyarrow>=15,<24"

python src/trendyol_cross_encoder.py \
  --mode train \
  --model-name "${MODEL_NAME}" \
  --train "${TRAIN_FILE}" \
  --data-dir "${ROOT}/data" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --experiment-id "${EXPERIMENT_ID}" \
  --max-length 160 \
  --batch-size 16 \
  --eval-batch-size 64 \
  --effective-batch-size 64 \
  --epochs 1.5 \
  --learning-rate 2e-5 \
  --eval-steps 500 \
  --dataloader-workers 4 \
  --early-stopping-patience 3 \
  --optim adamw_torch_fused \
  --gradient-checkpointing \
  --tf32 \
  --bf16 \
  --resume-checkpoint auto \
  --seed 42

SUMMARY="${ARTIFACTS_DIR}/experiments/${EXPERIMENT_ID}/train_summary.json"
MODEL_DIR="${ARTIFACTS_DIR}/experiments/${EXPERIMENT_ID}/model"
test -f "${SUMMARY}"
test -d "${MODEL_DIR}"

echo "Eğitim tamamlandı. Özet: ${SUMMARY}"
echo "Model: ${MODEL_DIR}"
cat "${SUMMARY}"
