# 0.901 → 0.94+ strateji ve durum (17 Temmuz 2026)

Metrik: **Macro-F1** (iki sınıfın F1 ortalaması). Yerelde güvenilir bir Macro-F1
proxy'si YOKTUR: resmi `training_pairs.csv`'nin tüm etiketleri `1`'dir (250k
pozitif), train/test `term_id` kesişimi sıfırdır, `train_testlike.parquet`'teki
`label=0` satırları madenlenmiş **pseudo-negatiflerdir** (weight 0.15,
"ambiguous"). Bu yüzden model seçimi LB dışında doğrulanamaz; strateji =
**bağımsız sinyaller + tutucu birleştirme + bilgi veren submission**.

## Doğrulanan durum

| Çapa | Pozitif | Oran | LB |
|---|---:|---:|---:|
| v6 (`00_proven_anchor_v6_lb0874.csv`) | 770.936 | 22.95% | 0.874 |
| consensus_medium (== `anchor_v6.parquet`) | 830.608 | 24.72% | **0.901** |

> Not: cascade içindeki `anchor_v6.parquet` ismi yanıltıcı — bu dosya aslında
> LB 0.901 alan `llm_consensus_medium`'dır (v6'dan 88.738 satır farklı). Tüm
> round-2 ve engine birleştirmeleri doğru şekilde 0.901 tabanı üzerinde yapıldı.

Round-2 cascade oyları yerelde mevcut: Qwen 180.000 (tüm havuz), Mistral 93.862
(Qwen'in güvenle flip istediği alt küme). **GPT-OSS henüz yok** (Colab bekliyor).
`qwen_mistral_strict` = 65.289 flip (60.063 0→1, 5.226 1→0) — kullanıcının
verdiği sayılarla birebir doğrulandı.

## Sinyal bağımsızlığı (0.901'de hâlâ 0 olan satırlar)

- **LLM round-2 (qwen∧mistral):** strict 60.063 0→1; medium 67.129 0→1. Proven
  0.874→0.901 sıçramasını üreten mekanizmanın aynısı. En yüksek soy, ama agresif.
- **Engine floor-deficit high-cert:** 4.703 0→1; %99.95 ≥2-sinyal yapısal
  uzlaşı; LLM'den bağımsız.
- **Forced-membership residual:** 141.179 fazla adayın %74'ü (104.621) 0.901'de
  ZATEN pozitif — "kaçırılan pozitif" hipotezini güçlü şekilde doğruluyor.
  Kalan 36.558 hâlâ negatif; temiz çekirdek (agreement-3, LLM-çelişkisi hariç)
  = 11.725, agreement≥2 = 23.112.

**Kritik uyarı:** LLM'in değerlendirdiği still-negatif residual'ların yalnız
%16'sını pozitif dedi. Yani yapı "force-added → pozitif" derken LLM çoğunlukla
"alakasız" diyor. Bu yüzden temiz yapısal küme LLM-negatif satırları dışlar
(kullanıcı kuralı: LLM ile açıkça çelişen yapısal kararı otomatik ekleme).

## GPT-OSS bulgusu ve düzeltmesi (Task 1)

Harmony final-channel parser'ı sağlam (9/9 test geçiyor). Ancak tam koşu
döngüsünde `parse_gpt_oss_final_label` **hata yakalama olmadan** çağrılıyordu:
`--gpt-oss-max-tokens 192` ile Harmony reasoning bütçesi dolup final kanal
yazılmazsa ValueError → tüm chunk çöker; temperature=0 deterministik olduğu için
yeniden koşu aynı satırda tekrar çöker (**muhtemelen GPT-OSS'un "bozulma"
nedeni**). Düzeltme: parse hatasında **abstain → çapayla aynı taraf** (flip'i
onaylamaz), fallback sayısı raporlanır, ilk chunk'ta >%20 ise erken durur.
Yamalı runner: `run_llm_judge_colab.py`; yeni paket:
`trendyol_llm_student_cascade_bundle_harmony_v4.zip` (input parquet'ler
değişmedi, checkpoint'ler korunur). `scripts/audit_gpt_oss.py` Colab oyları
gelince veto sayısını ve triple anlaşmayı raporlar.

## Üretilen adaylar (`artifacts/merged_candidates_v1/`, hiçbiri gönderilmedi)

| Dosya | flip | 0→1 | 1→0 | oran | hipotez |
|---|---:|---:|---:|---:|---|
| 01_llm_qwen_mistral_strict | 65.289 | 60.063 | 5.226 | 26.36% | LLM round-2 izole (güvenli) |
| 02_llm_qwen_mistral_medium | 73.693 | 67.129 | 6.564 | 26.53% | LLM round-2 izole (agresif) |
| 03_engine_floor_highcert | 4.703 | 4.703 | 0 | 24.86% | yapısal floor izole (en güvenli) |
| 04_engine_struct_3signal | 13.881 | 13.881 | 0 | 25.14% | yapısal (floor+residual a3) izole |
| 05_engine_struct_2signal | 24.126 | 24.126 | 0 | 25.44% | yapısal a≥2 izole |
| 06_combined_medium+3signal | 86.808 | 80.244 | 6.564 | 26.92% | tam yığın |

GPT-OSS gelince aynı `scripts/build_merged_candidates.py` otomatik olarak
GPT-vetolu (triple) sürümleri üretir.

## 0.94 hedefi — dürüst değerlendirme

İki kalibrasyon noktası: +59.672 net pozitif (round-1 LLM) = +0.027 Macro-F1.
Eldeki sinyaller (round-2 + yapısal) precision korunursa ~+0.01–0.02 daha
verebilir → gerçekçi hedef **~0.91–0.93**. **0.94 bir stretch**; büyük olasılıkla
Retrieval Replica V2'nin (şu an Recall@100 ~0.55, hedef ≥0.85) ve distilasyon
öğrencisinin gerçekten yeni yüksek-recall pozitifler üretmesini gerektirir — bu
işler Colab-ölçeğinde ve henüz kalite kapısında değil.

## Önerilen 3 submission (her biri bağımsız hipotez)

1. **LLM round-2** — `01_llm_qwen_mistral_strict` (GPT-OSS bitince GPT-vetolu
   `triple_strict` ile değiştir). Ana bahis.
2. **Yapısal bağımsız** — `04_engine_struct_3signal`. LLM'e dik; tutarsa yığılır.
3. **Tam yığın** — `06_combined_medium+3signal`. Yalnız 1 ve 2 pozitif hareket
   ederse denenmeli.

## Retrieval Replica V2 (Task 3) — Colab-hazır kod

`src/retrieval_replica_v2.py` (7 aşama: prepare→embed→baseline→mine→train→eval→
residual) + `colab/RETRIEVAL_REPLICA_V2_COLAB.py` sürücüsü. Hedef = gözlenen
retrieval-membership (relevance değil), query-disjoint split. Ölçüler:
Recall@100, Jaccard@100, p10 Recall@100; lexical + pretrained-dense baseline'a
karşı kazanç. **Recall@100 ≥ 0.85 kapısı** geçilmeden residual geniş ölçekte
kullanılmaz (kod bunu zorluyor). Yerelde MiniLM ile 7 aşama uçtan uca doğrulandı
(kapı mantığı dahil); gerçek koşu Colab'da TY-ecomm ile yapılır (yerelde ağırlık
`.incomplete`). n>100 grupları için `observed − predicted_top100` = forced-positive
sinyali parquet olarak üretilir.

## Query-cardinality (Task 5) — `artifacts/cardinality_v1/`

Bulgu: anchor'ın sorgu-başı k'si zaten iyi kalibre (student threshold-0.70 sayımı
ile korelasyon 0.98). Tek net kusur: **1.918 sorguda k=0** (her sorguda ≥1 alakalı
ürün olmalı). Relative-gap elbow sezgisi k'yi çok abartıyor (kullanılmadı). Üç aday
(anchor'ı sadece dürterek, kararlarını atmadan):

| aday | +ekle | −çıkar | oran | mantık |
|---|---:|---:|---:|---|
| card_conservative | 1.918 | 0 | 24.78% | yalnız k=0 tabanı (top-1) — en savunulabilir |
| card_medium | 7.013 | 0 | 24.93% | k=0 + 797 aşırı düşük-k sorgusunu student-desteğiyle yükselt |
| card_aggressive | 7.013 | 4.476 | 24.80% | + 146 aşırı yüksek-k sorgusunu kırp |

Train k-regressor çapraz kontrolü: val MAE 3.36, train k medyan 7 vs anchor 14 →
dağılım kayması doğrulandı; bu yüzden regressor seçici değil, yalnız raporlanır.

## Toplam aday envanteri (hepsi submission-valid, hiçbiri gönderilmedi)

`artifacts/merged_candidates_v1/` (7) + `artifacts/cardinality_v1/` (3). Tümü
`sample_submission.csv` ID sırasıyla birebir, yalnız 0/1.

Kaggle'a hiçbir dosya açık onay olmadan yüklenmez.
