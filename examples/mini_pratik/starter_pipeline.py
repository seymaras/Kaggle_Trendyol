from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42

# Bilgisayarda script'in bulunduğu klasörü kullanır.
# Kaggle'da gerekirse DATA_DIR yolunu /kaggle/input/... olarak değiştir.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = HERE

# 1) DOSYALARI OKU
terms = pd.read_csv(DATA_DIR / "terms.csv")
items = pd.read_csv(DATA_DIR / "items.csv")
positives = pd.read_csv(DATA_DIR / "training_pairs.csv")
hard_negatives = pd.read_csv(DATA_DIR / "manual_hard_negatives.csv")
test_pairs = pd.read_csv(DATA_DIR / "submission_pairs.csv")

print("Arama terimi:", len(terms))
print("Ürün:", len(items))
print("Pozitif eğitim çifti:", len(positives))


# 2) ID'LERİ GERÇEK METİNLERLE BİRLEŞTİR
def add_text_columns(pairs):
    """term_id ve item_id yerine gerçek arama/ürün bilgilerini getirir."""
    return (
        pairs.merge(terms, on="term_id", how="left")
        .merge(items, on="item_id", how="left")
        .fillna("")
    )


# 3) KOLAY NEGATİF ÜRET
# Her pozitif örnek için tamamen farklı ana kategoriden bir ürün seçiyoruz.
rng = np.random.default_rng(RANDOM_STATE)
item_category = items.set_index("item_id")["category"].to_dict()
positive_set = set(zip(positives.term_id, positives.item_id))
easy_rows = []

for number, row in enumerate(positives.itertuples(), start=1):
    positive_top_category = item_category[row.item_id].split("/")[0]
    candidates = items[
        items["category"].str.split("/").str[0].ne(positive_top_category)
        & ~items["item_id"].map(lambda item_id: (row.term_id, item_id) in positive_set)
    ]
    sampled_item = rng.choice(candidates["item_id"].to_numpy())
    easy_rows.append([f"EASY_{number:03d}", row.term_id, sampled_item, 0])

easy_negatives = pd.DataFrame(
    easy_rows, columns=["id", "term_id", "item_id", "label"]
)

# Pozitif + elle hazırlanan zor negatif + kodun ürettiği kolay negatif
train_pairs = pd.concat(
    [
        positives[["id", "term_id", "item_id", "label"]],
        hard_negatives[["id", "term_id", "item_id", "label"]],
        easy_negatives,
    ],
    ignore_index=True,
)

train = add_text_columns(train_pairs)
test = add_text_columns(test_pairs)
print(
    "Toplam eğitim satırı:", len(train),
    "| 1 sayısı:", int(train.label.sum()),
    "| 0 sayısı:", int((train.label == 0).sum()),
)


# 4) METİNLERİ BASİT SAYISAL ÖZELLİKLERE ÇEVİR
def words(text):
    """Metni küçük harfe çevirip kelime kümesine ayırır."""
    return set(re.findall(r"\w+", str(text).lower()))


def make_features(df):
    """Arama kelimelerinin ürün alanlarında ne kadar bulunduğunu ölçer."""
    feature_rows = []

    for row in df.itertuples():
        query_words = words(row.query)
        title_words = words(row.title)
        category_words = words(row.category)
        attribute_words = words(row.attributes)
        all_product_words = (
            title_words
            | category_words
            | attribute_words
            | words(row.gender)
            | words(row.age_group)
        )
        denominator = max(1, len(query_words))

        feature_rows.append(
            [
                len(query_words & title_words) / denominator,
                len(query_words & all_product_words) / denominator,
                len(query_words & category_words) / denominator,
                len(query_words & attribute_words) / denominator,
                len(query_words - title_words) / denominator,
            ]
        )

    return np.asarray(feature_rows, dtype=float)


X = make_features(train)
y = train["label"].to_numpy()


# 5) DOĞRULAMA: bazı arama terimlerini modelden tamamen sakla
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.33, random_state=RANDOM_STATE
)
train_indices, valid_indices = next(
    splitter.split(X, y, groups=train["term_id"])
)

model = LogisticRegression(
    max_iter=2000, class_weight="balanced", C=10, random_state=RANDOM_STATE
)
model.fit(X[train_indices], y[train_indices])
valid_probability = model.predict_proba(X[valid_indices])[:, 1]

# Model önce olasılık üretir. En iyi 0/1 karar sınırını doğrulama verisinde ararız.
best_threshold, best_score = 0.50, -1.0
for threshold in np.arange(0.20, 0.81, 0.02):
    valid_prediction = (valid_probability >= threshold).astype(int)
    score = f1_score(y[valid_indices], valid_prediction, average="macro")
    if score > best_score:
        best_threshold, best_score = float(threshold), float(score)

print(f"Doğrulama Macro F1: {best_score:.3f}")
print(f"Seçilen threshold: {best_threshold:.2f}")


# 6) BÜTÜN EĞİTİM VERİSİYLE MODELİ KUR VE SUBMISSION ÜRET
model.fit(X, y)
test_probability = model.predict_proba(make_features(test))[:, 1]
prediction = (test_probability >= best_threshold).astype(int)

submission = pd.DataFrame({"id": test["id"], "prediction": prediction})
submission.to_csv(DATA_DIR / "submission.csv", index=False)
print("submission.csv oluşturuldu:", len(submission), "satır")
print(submission.head())


# Gerçek yarışmada cevap anahtarı verilmez; Kaggle gizlice puanlar.
# Bu bölüm sadece oyuncak veriyle Kaggle puanını taklit eder.
answer_path = DATA_DIR / "practice_answer_key.csv"
if answer_path.exists():
    answer = pd.read_csv(answer_path)
    scored = submission.merge(answer, on="id")
    local_score = f1_score(
        scored["label"], scored["prediction"], average="macro"
    )
    print(f"Pratik Kaggle Macro F1 skoru: {local_score:.3f}")
