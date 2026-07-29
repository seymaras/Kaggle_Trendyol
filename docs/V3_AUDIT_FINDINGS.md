# V3 full-data audit findings

Bu rapor 11 Temmuz 2026 tarihinde `data/` altındaki gerçek beş CSV üzerinde
`trendyol_v3_core.py` ve query-only adversarial validation ile üretildi. Ham
raporlar `artifacts/v3/full_audit/` altındadır.

## Doğrulanan şema ve büyüklük

| Dosya | Satır | Ana kolonlar |
|---|---:|---|
| `items.csv` | 962.873 | `item_id,title,category,brand,gender,age_group,attributes` |
| `terms.csv` | 50.153 | `term_id,query` |
| `training_pairs.csv` | 250.000 | `id,term_id,item_id,label` |
| `submission_pairs.csv` | 3.359.679 | `id,term_id,item_id` |
| `sample_submission.csv` | 3.359.679 | `id,prediction` |

- Train etiketlerinin tamamı `1`.
- Train pair'lerinde 17.968 query ve 229.416 ürün var.
- Submission'da 32.185 query ve 929.781 ürün var.
- Train–test `term_id` kesişimi sıfır.
- Sample submission ID sırası submission pair ID sırasıyla birebir aynı.
- Duplicate train veya submission `(term_id,item_id)` çifti bulunmadı.

## Query-group dağılımı

| Dağılım | Medyan | p95 | Maksimum |
|---|---:|---:|---:|
| Train pozitif/query | 7 | 45 | 1.525 |
| Submission aday/query | 100 | 106 | 3.680 |

Bu fark, dengeli 1:1 validation Macro-F1'inin gerçek test karar problemini temsil
etmediğini doğruluyor. Query-uniform sampler kullanılmadığında 1.525 pozitifi olan
generic query'ler eğitimi de orantısız biçimde domine eder.

## Ürün alanları ve örtüşme

- Semantic missing: marka 4, gender 590.714, age group 572.028, attributes 19.025.
- Train ve submission ürün ID kesişimi 196.324; item Jaccard 0,2039.
- Kategori Jaccard 0,8492; marka Jaccard 0,4478.
- Normalize başlığı başka bir item ID ile aynı olan 83.173 ürün satırı ve 33.929
  exact başlık grubu bulundu.
- Geniş varyant-family proxy taraması ayrıca 134.707 satırı işaretledi. Bu sayı
  doğrudan label değildir; renk/beden/kapasite temizleme eşiği ablation ile
  doğrulanmalıdır.

Dolayısıyla aynı ürün ailesini güvenli negatif saymama koruması zorunlu; fakat ürün
ID overlap'ini tamamen purge etmek ana query-cold skor yanında ayrı bir stress test
olarak raporlanmalı.

## Query-only distribution shift

Word/char TF-IDF, uzunluk, sayı, birim ve Türkçe karakter özellikli 5-fold domain
classifier OOF AUC'si **0,5135** oldu. Fold AUC aralığı 0,5061–0,5298.

Bu sonuç train ve test query metinleri arasında güçlü, kolay ayrılabilir lexical
shift olmadığını gösteriyor. Eski cross-encoder validation 0,9327 → LB 0,863 farkının
ana şüphelileri şunlardır:

1. validation candidate/class-prior uyuşmazlığı;
2. sentetik false-negative gürültüsü;
3. checkpoint/threshold'un aynı validation üzerinde seçilmesi;
4. retrieval/candidate-generation shift'i;
5. olasılık kalibrasyonu ve query-group post-processing eksikliği.

Bu yüzden ilk GPU deneyi daha büyük model değil, sabit `validation_oracle` evreninde
Qwen/BGE outer OOF ve meta cross-fit calibration olmalıdır.
