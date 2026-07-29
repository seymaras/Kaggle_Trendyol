# Trendyol relevance pipeline v3

Bu sürümün amacı public leaderboard threshold’una uyum sağlamak değil; yeni sorgulara ve private leaderboard’a daha güvenilir biçimde genelleyen, bütün ham skorları saklayan bir OOF sistemi kurmaktır. `0.95+` hedefi bir aspirasyondur, garanti değildir.

## 1. En olası beş problem

1. **Validation evreni test evrenine benzemiyor.** Eski cross-encoder tek bir hash fold’da ve sentetik, çoğu dengeli negatiflerde ölçülüyordu. Gerçek testte 32.185 query ve 3.359.679 pair var; query başına medyan 100, p99 204, maksimum 3.680 aday bulunuyor. Dengeli 1:1 validation bu class prior’ı ve query-group kararını ölçmüyor. Outer validation ile checkpoint/epoch seçmek de OOF skorunu iyimser yapıyor.
2. **PU label noise “hard negative” adı altında büyüyor.** Train yalnız pozitif içeriyor. Yüksek cross-encoder veya teacher skorlu bilinmeyen adayları doğrudan `0` yapmak en muhtemel false-negative’leri seçiyor. Aynı normalize başlık, varyant ailesi, aşırı embedding benzerliği ve birden fazla modelin relevance mutabakatı ayrı `suspicious` havuzuna gitmeli.
3. **Qwen3 reranker yanlış model sınıfıyla açılabilir.** `Qwen/Qwen3-Reranker-0.6B` bir causal-LM reranker’dır. `AutoModelForSequenceClassification(..., ignore_mismatched_sizes=True)` pretrained scoring davranışını korumaz. Doğru skor resmi prompt’un son pozisyonundaki `yes_logit - no_logit` farkıdır. Bu uygulama `trendyol_v3_reranker.py` içindedir.
4. **Train–test query/candidate shift’i ve kalibrasyon eksikliği.** Train ve test term ID’leri kesişmiyor. Query-only adversarial validation; kelime, char n-gram, uzunluk, sayı, birim ve Türkçe karakter shift’ini ölçmeli. Threshold ve calibrator aynı OOF satırlarında fit edilip raporlanmamalı; ikinci bir meta cross-fit kullanılmalı.
5. **Tek ve yüksek korelasyonlu model ailesi ile stale artefakt riski.** Qwen, BGE/XLM-R ve Türkçe BERT farklı hata profilleriyle OOF üzerinde birleştirilmeli. Test shard’ı yalnız “dosya var” diye tekrar kullanılmamalı; ordered-ID hash, model fingerprint, satır sınırı ve config eşleşmeli. Logit ve probability hiçbir zaman silinmemeli.

## 2. Önerilen nihai mimari

```text
raw positives
  ├── Validation A: canonical-query-strict balanced Group split
  ├── Validation B: frozen query-embedding semantic-cluster split
  └── Validation C: query-only adversarial score + whole-cluster test-like holdout

outer-train only
  ├── catalog-only retrieval
  ├── Qwen3/BGE bi-encoder hard candidates
  ├── attribute-contradiction candidates
  └── family/model-ensemble false-negative triage
       ├── train negatives (typed + weighted)
       └── suspicious audit pool

fixed outer-validation candidate universe
  ├── natural: end-to-end retrieval + reranker
  └── oracle: reranker-only; missing known positives are inserted at fixed group size

models
  ├── A: Qwen3-Reranker-0.6B, QLoRA, yes/no logit, BCE + pairwise/ListNet
  ├── B: BAAI/bge-reranker-v2-m3, one-logit XLM-R, LoRA/full
  ├── C: Turkish BERT, 3 seeds, multi-sample dropout + optional FGM/SWA
  ├── bi-encoder cosine/retrieval score
  └── LightGBM/CatBoost feature model

complete outer OOF logits
  → meta cross-fit calibration
  → meta cross-fit threshold/policy
  → OOF-only ensemble/stacking
  → fold-average chunked test logits
  → validated binary submission
```

Model B için ilk tercih BGE’dir. `jina-reranker-v3` Qwen3 tabanlı özel listwise/late-interaction API kullanır, generic sequence-classifier değildir ve model kartı CC-BY-NC-4.0 lisansı belirtir. Yarışma kuralları ve lisans yorumu doğrulanmadan final ensemble’a eklenmemelidir.

## 3. Deneylerin öncelik sırası

1. A/B/C manifestleri, sabit validation candidate universe, tam OOF logit ve meta cross-fit threshold.
2. Eski negatifler → family filter → model-consensus filter → bi-encoder/attribute hard negatives ablation’ı.
3. Qwen3 0.6B tek fold DEBUG; resmi prompt, short/long text ve BCE/pairwise ağırlık ablation’ı.
4. Üç outer fold Qwen OOF; temperature/Platt/isotonic/beta cross-fit karşılaştırması.
5. BGE reranker ve Türkçe BERT seed ensemble; yalnız OOF korelasyonu yeterince düşükse ensemble’a katma.
6. 4B teacher’ı yalnız 100–300 bin seçilmiş positive/hard/uncertain/disagreement pair üzerinde çalıştırma; soft logit saklama.
7. OOF stacking, candidate-count/query-length bucket threshold ve query post-processing. OOF Macro-F1 artırmayan kural finalde kullanılmaz.

Her aşama önce `DEBUG=True` ile çalıştırılmalıdır. Public LB hiçbir ensemble ağırlığının veya threshold’un fit verisi değildir.

## 4. Tahmini GPU/RAM ihtiyacı

Değerler max length 192–256 ve tek GPU için yaklaşık başlangıç aralıklarıdır. Gerçek süreyi notebook’taki 4.096 pair benchmark’ı belirler.

| İş | VRAM | RAM/disk notu |
|---|---:|---|
| Qwen3 Embedding inference | 4–7 GB | 1024d item float16 cache ≈ 1,84 GiB; 256d ≈ 0,46 GiB |
| Qwen3 Embedding QLoRA | 6–10 GB | fold-specific cache çok pahalıdır |
| Qwen3 Reranker 0.6B QLoRA | 6–10 GB | T4’te fp16/SDPA; bf16 varsayılmaz |
| Qwen3 Reranker 0.6B full fp16 | 12–16+ GB | LoRA daha güvenli başlangıçtır |
| BGE reranker v2 M3 LoRA | 8–12 GB | XLM-R one-logit head korunur |
| Türkçe BERT full fine-tune | 3–6 GB | 3 seed ayrı checkpoint |
| 4B teacher QLoRA/inference | yaklaşık 14–24+ GB | yalnız seçilmiş pair’lerde kullanılır |

`estimated_hours = total_pairs × fold_count / pairs_per_second / 3600 × 1.15`. Tahmin 10 saati aşıyorsa 12 saatlik commit içinde o inference başlatılmaz; fold/model ayrı Kaggle commit’lerine bölünür.

## 5. Kod ve DEBUG çalıştırma

Ana notebook: `notebooks/06_relevance_v3.ipynb`

Modüller:

- `src/trendyol_v3_core.py`: şema, audit, normalizasyon, text view, intent ve contradiction.
- `src/trendyol_v3_validation.py`: A/B/C manifest, adversarial validation, metrik, calibration ve threshold.
- `src/trendyol_v3_mining.py`: fold-local curriculum mining ve false-negative triage.
- `src/trendyol_v3_biencoder.py`: multi-positive InfoNCE, memmap embedding ve FAISS mining.
- `src/trendyol_v3_reranker.py`: Qwen yes/no ile generic one-logit training/inference.
- `src/trendyol_v3_ensemble.py`: OOF ensemble, experiment/ablation logu ve submission.
- `src/trendyol_v3_pipeline.py`: stage orkestrasyonu.

CPU smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 python src/trendyol_v3_pipeline.py \
  --debug \
  --debug-items 5000 \
  --debug-train-pairs 5000 \
  --debug-submission-pairs 5000 \
  --n-splits 3 \
  --semantic-clusters 48 \
  --embedding-backend lexical \
  --output-dir artifacts/v3/debug_run
```

Tam hazırlıkta semantic split için frozen Qwen embedding:

```bash
python src/trendyol_v3_pipeline.py \
  --no-debug \
  --n-splits 3 \
  --semantic-clusters 192 \
  --embedding-backend qwen \
  --negative-ratio 2 \
  --retrieval-top-k 300 \
  --output-dir artifacts/v3/full_run
```

## 6. Validation ve threshold raporlama

`validation_oracle.parquet` reranker-only, `validation_natural.parquet` retrieval dahil end-to-end rapordur. Unknown pair’lerin label kaynağı `unobserved_assumed_negative` olduğu için bu skor silver/PU metriktir; elle etiketli gold audit ayrıca tutulmalıdır.

Her OOF dosyası en az şu alanları içerir:

```text
pair_uid, term_id, item_id, label, fold, seed, model_key,
raw_score_kind, raw_logit, raw_probability
```

Kalibrasyon ve threshold aynı fold üzerinde fit+rapor edilmez:

```python
from trendyol_v3_validation import cross_fit_calibration_and_threshold

calibrated_oof, report = cross_fit_calibration_and_threshold(
    oof,
    methods=("none", "temperature", "platt", "isotonic", "beta"),
    seed=42,
)
calibrated_oof.to_parquet("artifacts/v3/calibrated_oof.parquet", index=False)
report.to_csv("artifacts/v3/calibration_report.csv", index=False)
```

Rapor TN/FP/FN/TP, class-0/1 precision/recall/F1, Macro-F1, positive rate, log-loss, Brier ve ECE içerir. Candidate-count bucket’ları görülmeyen veya düşük destekli bucket’larda global threshold’a döner.

## 7. Hard-negative mining

Her negatifte tip, hardness, fold, epoch, manifest hash, generator config hash, miner model hash ve miner train-term hash bulunur. Epoch 0 kolay/orta; epoch 2 bütün hard türleri açar.

```bash
python src/trendyol_v3_mining.py \
  --data-dir data \
  --manifest artifacts/v3/full_run/validation/group_manifest.parquet \
  --manifest-meta artifacts/v3/full_run/validation/group_manifest.json \
  --fold 0 \
  --epoch 2 \
  --negative-ratio 2 \
  --external-candidates artifacts/v3/biencoder/bi_encoder_hard_negatives.parquet \
  --output-dir artifacts/v3/full_run/mined_fold0 \
  --no-debug
```

Yüksek bi-encoder skoru + açık attribute contradiction güvenilir hard-negative olabilir. Yüksek cross-encoder + teacher + ensemble relevance mutabakatı, aynı ürün ailesi veya çok yüksek embedding/no-contradiction satırı `suspicious` olur; varsayılan eğitimden çıkarılır.

## 8. Model eğitimi

Qwen doğru causal yes/no yoluyla:

```bash
python src/trendyol_v3_reranker.py \
  --mode train \
  --input artifacts/v3/full_run/folds/fold_0/epoch_2_train.parquet \
  --model-name Qwen/Qwen3-Reranker-0.6B \
  --architecture qwen_causal \
  --model-key qwen3_06b_fold0 \
  --epochs 2 \
  --batch-size 2 \
  --effective-batch-size 32 \
  --max-length 256 \
  --lora --qlora \
  --output-dir artifacts/v3/models/qwen_fold0
```

BGE farklı mimari model:

```bash
python src/trendyol_v3_reranker.py \
  --mode train \
  --input artifacts/v3/full_run/folds/fold_0/epoch_2_train.parquet \
  --model-name BAAI/bge-reranker-v2-m3 \
  --architecture sequence_classifier \
  --model-key bge_m3_fold0 \
  --epochs 2 \
  --batch-size 2 \
  --effective-batch-size 32 \
  --max-length 256 \
  --lora --no-qlora \
  --output-dir artifacts/v3/models/bge_fold0
```

Outer validation checkpoint seçmek için kullanılmaz. Epoch sayısı ayrı inner pilotta kilitlenir; sonra outer-train’in tamamı aynı sabit epoch ile eğitilip outer-validation yalnız bir kez skorlanır. `TrainConfig` BCE/focal, pairwise, ListNet, distillation, auxiliary heads, multi-sample dropout, FGM ve SWA ablation’larını destekler.

## 9. Chunked test inference

```bash
python src/trendyol_v3_reranker.py \
  --mode infer \
  --input data/submission_pairs.csv \
  --data-dir data \
  --model-path artifacts/v3/models/qwen_fold0/final \
  --model-name Qwen/Qwen3-Reranker-0.6B \
  --architecture qwen_causal \
  --model-key qwen3_06b_fold0 \
  --inference-batch-size 32 \
  --shard-size 50000 \
  --item-view-cache artifacts/v3/cache/item_view_long.parquet \
  --output-dir artifacts/v3/test_scores/qwen_fold0
```

Her shard `_row_idx,id,term_id,item_id,raw_logit,probability_raw,model_key` saklar. OOM olduğunda aynı satır aralığı batch yarıya indirilerek tekrar skorlanır. Shard reuse ordered-ID hash ve model fingerprint ile doğrulanır. Final Parquet shard’lar RAM’de concat edilmeden PyArrow writer ile birleştirilir.

## 10. Ensemble ve calibration

```bash
python src/trendyol_v3_ensemble.py \
  --mode fit \
  --oof-scores \
    qwen=artifacts/v3/oof/qwen.parquet \
    bge=artifacts/v3/oof/bge.parquet \
    trbert=artifacts/v3/oof/trbert_seedavg.parquet \
    lgbm=artifacts/v3/oof/lgbm.parquet \
  --test-scores \
    qwen=artifacts/v3/test/qwen_foldavg.parquet \
    bge=artifacts/v3/test/bge_foldavg.parquet \
    trbert=artifacts/v3/test/trbert_seed_foldavg.parquet \
    lgbm=artifacts/v3/test/lgbm.parquet \
  --method logistic \
  --calibration temperature \
  --output-dir artifacts/v3/ensemble
```

Araç average, constrained weighted average, query-local rank average, geometric mean ve logistic stacking’i fold-dışı OOF ile karşılaştırır. Spearman correlation matrisi yazılır. Candidate universe farklıysa ensemble durur.

## 11. Final submission üretme ve doğrulama

Threshold yalnız meta cross-fit OOF raporundan gelir:

```bash
python src/trendyol_v3_ensemble.py \
  --mode submit \
  --scores artifacts/v3/ensemble/ensemble_test_probabilities.parquet \
  --threshold 0.517 \
  --sample-submission data/sample_submission.csv \
  --submission-pairs data/submission_pairs.csv \
  --output-dir artifacts/v3/final
```

`0.517` yalnız örnek komut değeridir; gerçek değer `ensemble_parameters.json` içindeki OOF-selected threshold olmalıdır. Kod satır sayısını, unique ID’leri, sample ID sırasını, eksik skorları ve yalnız `0/1` prediction bulunduğunu doğrular. `final_submission_probabilities.parquet`, binary CSV’den bağımsız olarak raw logit ve probability’leri korur; summary JSON ilk/son beş satırı ve pozitif oranını yazar.

## Deney ve ablation zorunluluğu

`ExperimentLogger` istenen 27 kolonu hem CSV hem JSON’a yazar. Ablation sırası:

```text
old negatives → family-filtered → dynamic hard negatives → field text → query normalization
→ pairwise/ListNet → Qwen → BGE → teacher → calibration → query policy → ensemble
```

Her satırda önceki Macro-F1, yeni Macro-F1 ve fark saklanır. Kazanç `0.001–0.002` iken inference süresi 1,5×’ten fazla artarsa `low_roi_warning=True` olur.

## Kaynak sözleşmeleri

- Qwen3 reranker resmi model kartı: <https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>
- Qwen3 embedding resmi model kartı: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>
- BGE reranker resmi model kartı: <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- Jina v3 model kartı ve lisansı: <https://huggingface.co/jinaai/jina-reranker-v3>

Yarışma kuralları bu depoda bulunmadığı için dış veri, remote API ve model lisansı uygunluğu varsayılmamıştır. Final commit öncesinde model/dataset lisansları ve internet/dış veri kuralları yarışmanın güncel resmi metniyle ayrıca doğrulanmalıdır.
