# Trendyol negatif örnekleme notebook’u

## İlk deneme

1. Kaggle’da yeni bir Notebook açın.
2. Yarışma verisini **Add Input** ile notebook’a ekleyin.
3. `trendyol_negative_sampling_kaggle.ipynb` dosyasını import edin.
4. Config hücresinde `DATA_DIR` yolunu Kaggle’daki yarışma klasörüne göre değiştirin. Yol yanlışsa `AUTO_FIND_FILES=True` dosyaları otomatik arar.
5. İlk çalıştırmada ayarları değiştirmeyin:
   - `USE_FULL_DATA = False`
   - `POS_SAMPLE_SIZE = 50_000`
   - `USE_FULL_TFIDF_CATALOG = False`
6. Hücreleri yukarıdan aşağı sırayla çalıştırın.

## Beklenen çıktı

Notebook şu dosyayı her durumda üretir:

`/kaggle/working/train_with_negatives.csv`

Ortamda Parquet motoru varsa ayrıca şunu üretir:

`/kaggle/working/train_with_negatives.parquet`

## Tam veriye geçiş

Küçük örnek başarıyla tamamlandıktan sonra:

```python
USE_FULL_DATA = True
```

TF-IDF bölümünde bellek sorunu yaşarsanız `TFIDF_ITEM_POOL_SIZE` değerini düşürün. Bütün kataloğu TF-IDF havuzuna almak isterseniz ayrıca `USE_FULL_TFIDF_CATALOG=True` yapabilirsiniz; bu seçenek güçlü makine gerektirebilir.

## Kritik uyarı

Üretilen hard-negative’ler kesin gerçek negatif değildir. Eğitimde görünmeyen alakalı ürünler bulunabilir. Özellikle `title_similar_hard` ve `same_top_category_hard` örneklerinden rastgele bir grup gözle kontrol edilmelidir.
