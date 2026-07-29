# TEKNOFEST 2026 Trendyol E-Ticaret Datathon

Arama terimi–ürün çiftlerinin alakalı (`1`) veya alakasız (`0`) olduğunu tahmin etme projesi.

## Proje yapısı

```text
Kaggle_Trendyol/
├── notebooks/   # Kaggle/Jupyter notebook'ları
├── src/         # Notebook'un düz Python sürümü
├── data/        # Yarışma CSV'leri; Git'e eklenmez
├── artifacts/   # Üretilen eğitim verileri ve modeller; Git'e eklenmez
├── examples/    # Küçük yapay pratik veri seti
├── docs/        # Kullanım notları
└── requirements.txt
```

## Yerel ortamı açma

```bash
cd /Users/seyma/Documents/Kaggle_Trendyol
source .venv/bin/activate
jupyter-lab
```

Tarayıcı açıldığında `notebooks/trendyol_negative_sampling_kaggle.ipynb` dosyasını seçin.
Notebook kernel olarak **Python (Kaggle Trendyol)** kullanacak şekilde ayarlanmıştır.

## Pipeline v2

Query-driven hard-negative mining, dürüst retrieval validasyonu, LightGBM, GPU cross-encoder ve ensemble akışı için [PIPELINE_V2.md](docs/PIPELINE_V2.md) belgesini izleyin.

Hızlı kontrol:

```bash
make test
make smoke
```

## Relevance pipeline v3

Query-cold/semantic/test-like validation, fold-local false-negative filtreleme,
Qwen3/BGE reranker, bi-encoder mining, cross-fitted calibration ve güvenli
submission akışı için [PIPELINE_V3.md](docs/PIPELINE_V3.md) belgesini ve
`notebooks/06_relevance_v3.ipynb` notebook'unu kullanın.
Gerçek CSV'lerde ölçülen ilk bulgular [V3_AUDIT_FINDINGS.md](docs/V3_AUDIT_FINDINGS.md)
dosyasındadır.

İlk uçtan uca CPU smoke koşusu:

```bash
python src/trendyol_v3_pipeline.py \
  --debug --debug-train-pairs 5000 --debug-submission-pairs 5000 \
  --output-dir artifacts/v3/debug_run
```

## Gerçek yarışma verisi

Şu dosyaları `data/` klasörüne koyabilirsiniz:

- `items.csv`
- `terms.csv`
- `training_pairs.csv`
- `submission_pairs.csv`
- `sample_submission.csv`

Yarışma verisi ve işlenmiş kopyaları Git'e eklenmez. Notebook'ta yerel kullanım için:

```python
DATA_DIR = Path("../data")
```

Kaggle üzerinde ise `DATA_DIR`, **Add Input** ile eklenen dataset klasörünü göstermelidir.

## İlk güvenli çalışma

Önce şu ayarlarla küçük örnek üzerinde çalışın:

```python
USE_FULL_DATA = False
POS_SAMPLE_SIZE = 50_000
USE_FULL_TFIDF_CATALOG = False
```

Her şey tamamlandıktan sonra `USE_FULL_DATA = True` yapabilirsiniz.
