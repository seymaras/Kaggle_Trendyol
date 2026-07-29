# Trendyol relevance pipeline v2

`src/` tek kaynak kodudur. Notebook'lar yalnızca Kaggle ortamında ilgili script'i çağırır. Bütün yeni çıktılar `artifacts/experiments/<experiment_id>/` altında tutulur; eski baseline dosyaları değiştirilmez.

## 1. Test ve hızlı baseline

```bash
make test
EXPERIMENT_ID=baseline_v2 .venv/bin/python src/trendyol_train_baseline.py
.venv/bin/python src/make_ratio_submissions.py \
  --probabilities artifacts/experiments/baseline_v2/probabilities.parquet \
  --sample-submission data/sample_submission.csv \
  --output-dir artifacts/experiments/baseline_v2
```

## 2. Query-driven negatifler ve retrieval raporu

Önce smoke test:

```bash
make smoke
```

Tam veri:

```bash
make mine
.venv/bin/python src/trendyol_validation.py \
  --retrieval-candidates artifacts/experiments/query_mining_v2/retrieval_candidates.parquet \
  --output-dir artifacts/experiments/query_mining_v2
```

Miner word ve char TF-IDF ile katalogdaki ilk 300 adayı birleştirir; bilinen pozitifleri çıkarır. Negatif hedefi query'deki pozitif sayısına göre `clip(2 × pozitif, 12, 40)` olarak belirlenir. Generic query'lerde top-10 ambiguous aday kullanılmaz; toplam negatif sample weight varsayılan olarak pozitif weight toplamına eşitlenir.

## 3. Classical v2 ve gold audit

```bash
make install-gpu
.venv/bin/python src/trendyol_train_classical_v2.py \
  --mode train \
  --train artifacts/experiments/query_mining_v2/train_query_negatives.parquet \
  --experiment-id classical_v3 \
  --tfidf-max-features 20000

.venv/bin/python src/trendyol_train_classical_v2.py \
  --mode infer \
  --experiment-id classical_v3

.venv/bin/python src/prepare_gold_audit.py \
  --probabilities artifacts/experiments/classical_v3/probabilities.parquet \
  --output artifacts/gold_audit_2000.csv
```

`human_label` alanını `0/1`, emin olunmayan satırlarda `uncertain=1` olarak doldurun. `audit_split=final` satırları model/threshold seçerken kullanılmaz.

Her model için aynı gold raporunu üretin:

```bash
python src/evaluate_gold.py \
  --probabilities artifacts/experiments/classical_v3/probabilities.parquet \
  --gold artifacts/gold_audit_2000.csv \
  --output artifacts/experiments/classical_v3/gold_report.csv
```

## 4. Cross-encoder

Kaggle güçlü GPU ortamında `notebooks/05_cross_encoder.ipynb` çalıştırılabilir. Eşdeğer komut:

```bash
python src/trendyol_cross_encoder.py \
  --mode train-infer \
  --model-name BAAI/bge-reranker-v2-m3 \
  --train artifacts/experiments/query_mining_v2/train_query_negatives.parquet \
  --experiment-id bge_cross_encoder --bf16
```

XLM-R diversity deneyi için `--model-name FacebookAI/xlm-roberta-large --experiment-id xlmr_cross_encoder` kullanın. Inference shard'ları mevcutsa yeniden hesaplanmaz.

İkinci tur negatif:

```bash
python src/trendyol_cross_encoder.py \
  --mode infer \
  --model-path artifacts/experiments/bge_cross_encoder/model \
  --pairs artifacts/experiments/query_mining_v2/retrieval_candidates.parquet \
  --experiment-id bge_train_candidates

python src/mine_model_hard_negatives.py \
  --probabilities artifacts/experiments/bge_train_candidates/probabilities.parquet \
  --output artifacts/experiments/bge_cross_encoder/model_hard_negatives.parquet

python src/combine_training_data.py \
  --base artifacts/experiments/query_mining_v2/train_query_negatives.parquet \
  --extra artifacts/experiments/bge_cross_encoder/model_hard_negatives.parquet \
  --output artifacts/experiments/bge_cross_encoder/train_stage2.parquet
```

Ardından BGE'yi birleşik dosyada `--epochs 0.5 --resume-checkpoint <checkpoint>` ile devam ettirin.

## 5. Query-grouped CatBoost ranker

Test-like adayları yeniden üretip bilinmeyen retrieval adaylarını confidence-weighted hard negative olarak eğitmek için:

```bash
make testlike
make ranker
```

`build_testlike_training.py` top-10 adayları kesin negatif saymaz: bilinen pozitifler `1.0`, rank 1–10 adayları `0.15`, rank 11–50 adayları `0.50`, daha aşağı adaylar `0.85` ağırlık alır. Böylece hard-negative öğrenilirken olası false-negative etkisi azaltılır.

Ranker eğitimi sonunda `cv_report.csv` içinde fold ve global OOF `macro_f1/threshold`, ayrıca `oof_probabilities.parquet` içinde raw score ve probability bulunur.

Eğitimden sonra ranker inference:

```bash
make test-retrieval

python src/trendyol_train_catboost_ranker.py \
  --mode infer \
  --experiment-id catboost_ranker_v1 \
  --test-retrieval artifacts/test_retrieval.parquet \
  --task-type GPU

python src/make_submission_from_probs.py \
  --probabilities artifacts/experiments/catboost_ranker_v1/probabilities.parquet \
  --mode ratio --positive-rate 0.23 \
  --output artifacts/experiments/catboost_ranker_v1/submission_ratio_023.csv
```

İlk ranker deneyinde önce `YetiRankPairwise` ve `PairLogit` aynı split üzerinde karşılaştırılmalı; Optuna yalnız daha iyi loss seçildikten sonra çalıştırılmalı.

Seçilen loss için tuner:

```bash
python src/tune_catboost_ranker.py \
  --train artifacts/train_testlike.parquet \
  --trials 20 --task-type GPU \
  --output artifacts/experiments/catboost_ranker_v1/optuna_best.json
```

## 6. Ensemble

Gold dosyasında en az 1.800 kesin etiket tamamlandıktan sonra:

```bash
python src/trendyol_ensemble.py \
  --scores bge=artifacts/experiments/bge_cross_encoder/probabilities.parquet \
           xlmr=artifacts/experiments/xlmr_cross_encoder/probabilities.parquet \
           lgbm=artifacts/experiments/classical_v3/probabilities.parquet \
  --gold artifacts/gold_audit_2000.csv \
  --experiment-id ensemble_v1
```

Araç temperature calibration, ağırlık grid search, gold-final kapısı, query-level calibrator ve %23/%24/%25 submission'larını üretir.

## Güvenlik kontrolleri

- Full submission üretimi öncesinde id/sıra `sample_submission.csv` ile doğrulanır.
- Probability dosyalarında duplicate, NaN, Inf ve `[0,1]` dışı değer kabul edilmez.
- Fold ataması term id hash'iyle deterministiktir.
- Known-positive collision miner tarafından fatal hata olarak değerlendirilir.
