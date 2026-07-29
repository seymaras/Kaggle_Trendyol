# LLM Student Cascade

Bu paket Kaggle'a otomatik submission göndermez.

## Amaç

LB 0.901 alan `llm_consensus_medium.csv` yeni çapa kabul edilir. İlk turdaki
386.712 Qwen ve 124.244 Mistral kararı, 54 bağımsız/türetilmiş test özelliğine
öğrenci olarak damıtılır. Sorgu bazında tamamen ayrılmış doğrulama sonucu:

- AUC: 0.881145
- Average precision: 0.930238
- F1 @ 0.5: 0.855843

Öğrenci doğrudan submission üretmez. Yalnız 0.901 çapasına itiraz eden yüksek
bilgi değerli 180.000 satırı seçer. Bu satırlar Colab A100 üzerinde üç aşamalı
hakem zincirinden geçer:

1. Qwen3-30B-A3B tüm havuz
2. Mistral Small 24B yalnız Qwen'in güçlü itirazları
3. GPT-OSS 20B yalnız ilk iki modelin uzlaştığı çekirdek

## Colab

1. Runtime olarak A100 80 GB seçin.
2. `trendyol_llm_student_cascade_bundle.zip` dosyasını `/content` içine yükleyin.
3. `LLM_OGRENCI_CASCADE_TEK_HUCRE.py` içeriğini tek hücreye yapıştırıp çalıştırın.
4. Bağlantı kesilirse aynı hücreyi yeniden çalıştırın. Tamamlanan Parquet
   checkpoint'leri Drive'dan doğrulanıp atlanır.
5. Bittiğinde önce
   `MyDrive/TrendyolLLMStudentCascade/output/llm_judge_report.json` dosyasını
   paylaşın. CSV seçimi rapordaki flip sayıları ve model anlaşmasına göre yapılır.

## Güvenlik

- Paket ve giriş Parquet'leri SHA-256 ile doğrulanır.
- Her model ayrı subprocess'te çalışır; GPU belleği aşamalar arasında boşalır.
- Model revision'ları ilk çalışmada sabitlenir; resume aynı revision ile devam eder.
- Çıktı ID sırası ve ikili prediction değerleri finalizasyonda doğrulanır.
- Kaggle API çağrısı yoktur.
