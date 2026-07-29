# Trendyol engine-replica akışı

Bu akış yarışmaya submission göndermez. Amacı, `submission_pairs.csv` içindeki
aday-kümesi yapısını ayrı bir gözlenebilir hedef olarak öğrenmek ve herhangi bir
anchor submission üzerine uygulanabilecek yapısal residual üretmektir.

## Hipotez ve ölçülebilir kapı

- 32.185 test sorgusunun 30.378'i tam 100 aday içeriyor.
- 1.807 sorguda 100'ün üzerinde toplam 141.179 satır bulunuyor.
- Tam-100 gruplar hidden retrieval motorunun gözlenmiş adayları olarak kullanılır.
- Yakın sorguların adayları hard decoy olur; query-cold holdout'ta membership
  `Recall@100` raporlanır.
- CatBoost replica, dense ve lexical tekil kanallar aynı query-cold holdout'ta
  karşılaştırılır; en yüksek `Recall@100` otomatik olarak top-100 sınırını
  belirler. Üç kanal residual güveni için bağımsız oy olarak korunur.

## Yerel gerçek-veri smoke

```bash
python src/build_engine_replica_sample.py \
  --exact-terms 150 --excess-terms 25 \
  --output-dir artifacts/engine_replica_real_sample/data

python src/build_trendyol_domain_features.py \
  --stage all \
  --data-dir artifacts/engine_replica_real_sample/data \
  --output-dir artifacts/engine_replica_real_sample/domain \
  --pairs artifacts/engine_replica_real_sample/data/submission_pairs.csv \
  --feature-output artifacts/engine_replica_real_sample/submission_domain_features.parquet \
  --model-name Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 \
  --dimension 128 --batch-size 32 --device cpu

python src/trendyol_engine_replica.py \
  --mode all \
  --features artifacts/engine_replica_real_sample/submission_domain_features.parquet \
  --cache-dir artifacts/engine_replica_real_sample/domain \
  --output-dir artifacts/engine_replica_real_sample/replica \
  --max-exact-terms 150 --iterations 120
```

## Tam GPU koşusu

`kaggle_engine_dataset/` özel Kaggle dataset girdisini,
`kaggle_engine_kernel/` ise GPU script kernel'ini içerir. Kernel metadata'sında
competition kaynağı yoktur ve kod hiçbir Kaggle submission çağrısı yapmaz.

Kalıcı çıktılar:

- `engine_replica_report.json`: query-cold membership metriği;
- `structural_residual_candidates.parquet`: 141.179 beklenen residual ve güven
  kolonları;
- `structural_residual_summary.json`: adet/agreement kontrolleri;
- `fallback_anchor_submission_variants.zip`: yalnız 0→1 değişiklikli yerel
  örnekler.

## En iyi anchor'ı sonradan uygulama

En iyi anchor dosyası modele girdi değildir. Residual bir kez üretildikten sonra:

```bash
python src/apply_engine_residual.py \
  --anchor /path/to/anchored_fix_v6_final.csv \
  --residual artifacts/engine_replica/structural_residual_candidates.parquet \
  --sample data/sample_submission.csv \
  --output-dir artifacts/engine_replica/final_anchor_variants
```

Üretilen hiçbir CSV otomatik yüklenmez.

## GPU beklemeden test pseudo-pozitifleri

Mevcut classical feature shard'ları hızlı lexical motor vekilini üretir. Gerçek
örneklemde resmî akışla sıra korelasyonu 0,984 ve residual Jaccard'ı 0,932'dir.

```bash
python src/build_fast_structural_pseudolabels.py
```

`test_structural_pseudopositives.parquet` test dağılımından gelen yapısal
pozitifleri üç güven katmanında ve eğitim için `sample_weight` ile saklar.

Her sorgunun dosyada görülme sırası ayrıca gizli aday pozisyonu olarak korunur.
Training retrieval havuzundaki rank prior'ı ile ayrı ablation CSV'leri üretmek
için:

```bash
python src/build_position_prior_variants.py
```

Bu komut da yalnız yerel CSV/parquet üretir; Kaggle API çağrısı yapmaz.
