# Mini Trendyol–Kaggle Pratik Projesi

Bu klasör gerçek yarışma yapısını taklit eden küçük ve tamamen yapay bir veri setidir.

## Dosyalar

- `terms.csv`: Arama terimleri.
- `items.csv`: Ürün kataloğu.
- `training_pairs.csv`: Sadece pozitif (1) eğitim çiftleri.
- `manual_hard_negatives.csv`: Bizim elle ürettiğimiz zor negatif örnekler.
- `submission_pairs.csv`: Tahmin yapılacak çiftler.
- `sample_submission.csv`: Kaggle'a yüklenecek dosyanın biçimi.
- `practice_answer_key.csv`: Yalnızca pratikte yerel Kaggle puanını görmek için cevap anahtarı. Gerçek yarışmada bu dosya verilmez.
- `trendyol_kaggle_baslangic.ipynb`: Kaggle'a yükleyip adım adım çalıştıracağınız notebook.
- `starter_pipeline.py`: Notebook ile aynı mantığın tek dosyalık çalışan sürümü.

## Kaggle'da deneme

1. Kaggle'da yeni bir Notebook açın.
2. Bu klasördeki CSV dosyalarını bir Dataset olarak yükleyin veya notebook'a ekleyin.
3. `trendyol_kaggle_baslangic.ipynb` dosyasını import edin.
4. Hücreleri sırayla çalıştırın.
5. Son hücre `submission.csv` oluşturur.

İlk hedef yüksek puan değil; veri okuma → negatif üretme → model → tahmin → submission akışını bir kez tamamlamaktır.
