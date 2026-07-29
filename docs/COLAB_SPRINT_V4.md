# BERTurk v4 — Colab A100 eğitim akışı

Bu akış yalnız eğitim içindir. Kaggle submission göndermez ve mevcut 0.867
submission'ına dokunmaz. Girdi, term-gruplu sabit split içeren 828.002 satırlık
retrieval-eşleşmeli parquet'tir.

## 1. Colab ayarı

Colab'da `Runtime > Change runtime type > A100 GPU` seç. Ardından aşağıdaki
hücreyi çalıştır ve `Kaggle_Trendyol_sprint_v4_train.zip` dosyasını yükle:

```python
from google.colab import files, drive
uploaded = files.upload()
drive.mount('/content/drive')

!mkdir -p /content/Kaggle_Trendyol /content/empty_data
!unzip -oq /content/Kaggle_Trendyol_sprint_v4_train.zip -d /content/Kaggle_Trendyol
%cd /content/Kaggle_Trendyol
!nvidia-smi
!pip install -q "torch>=2.2,<3" "transformers>=4.51.3,<5" "accelerate>=1.0,<2" "pyarrow>=15,<24" sentencepiece
```

`nvidia-smi` çıktısında A100 görünmüyorsa tam eğitimi başlatma.

## 2. Zorunlu 20k profil koşusu

```python
!python src/trendyol_cross_encoder.py \
  --mode train \
  --model-name dbmdz/bert-base-turkish-cased \
  --train artifacts/sprint_v4/ce_train_retrieval_matched.parquet \
  --data-dir /content/empty_data \
  --artifacts-dir /content/profile_outputs \
  --experiment-id berturk_v4_profile20k \
  --no-refresh-catalog-text \
  --max-train-rows 20000 \
  --max-length 128 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --effective-batch-size 64 \
  --epochs 1 \
  --learning-rate 3e-5 \
  --eval-steps 100 \
  --dataloader-workers 2 \
  --early-stopping-patience 3 \
  --gradient-checkpointing \
  --tf32 --bf16 --no-fp16 \
  --seed 42

!cat /content/profile_outputs/experiments/berturk_v4_profile20k/train_summary.json
```

Koşu sorunsuz bitmeli; `validation_macro_f1`, `validation_threshold`,
`validation_positive_rate` ve `train_runtime` değerleri oluşmalıdır. OOM olursa
batch size'ı 32 yapıp effective batch size'ı 64 bırak.

## 3. Tam eğitim

Profil başarılıysa aşağıdaki hücreyi çalıştır. Checkpoint ve model Drive'a
yazılır; Colab koparsa aynı hücre `--resume-checkpoint auto` ile devam eder.

```python
!mkdir -p /content/drive/MyDrive/trendyol_sprint_v4_outputs

!python src/trendyol_cross_encoder.py \
  --mode train \
  --model-name dbmdz/bert-base-turkish-cased \
  --train artifacts/sprint_v4/ce_train_retrieval_matched.parquet \
  --data-dir /content/empty_data \
  --artifacts-dir /content/drive/MyDrive/trendyol_sprint_v4_outputs \
  --experiment-id berturk_v4_full \
  --no-refresh-catalog-text \
  --max-length 128 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --effective-batch-size 64 \
  --epochs 2 \
  --learning-rate 3e-5 \
  --eval-steps 1000 \
  --dataloader-workers 2 \
  --early-stopping-patience 3 \
  --gradient-checkpointing \
  --tf32 --bf16 --no-fp16 \
  --resume-checkpoint auto \
  --seed 42

!cat /content/drive/MyDrive/trendyol_sprint_v4_outputs/experiments/berturk_v4_full/train_summary.json
```

## 4. Sonuçlar

Drive altında şu dosyalar kalıcı olmalıdır:

- `model/`: seçilen en iyi checkpoint
- `checkpoints/`: kopma halinde devam dosyaları
- `train_summary.json`: Macro-F1, eşik, pozitif oran ve süre
- `validation_probabilities.parquet`: sonraki hata analizi/eşikleme girdisi
- `config.json`: yeniden üretilebilir eğitim ayarları

Tam eğitim tamamlanınca `train_summary.json` içeriğini Codex'e gönder. Sonraki
kapı, retrieval-validasyon sonucu üzerinden inference'a geçip geçmemektir.
