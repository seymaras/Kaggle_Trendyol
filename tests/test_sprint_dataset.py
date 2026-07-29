import pandas as pd

from build_sprint_dataset import add_query_views, grouped_holdout, label_and_triage, sample_training_rows


def test_group_holdout_is_term_disjoint_and_reproducible():
    terms = pd.Series([f"t{i}" for i in range(20) for _ in range(3)])
    one = grouped_holdout(terms, valid_size=0.2, seed=42)
    two = grouped_holdout(terms, valid_size=0.2, seed=42)
    assert one.equals(two)
    assert one.term_id.is_unique
    assert set(one.split) == {"train", "valid"}


def test_pu_triage_excludes_high_risk_unknowns_and_keeps_positives():
    candidates = pd.DataFrame({
        "term_id": ["t1"] * 6, "item_id": [f"i{i}" for i in range(6)], "query": ["IŞIK"] * 6,
        "retrieval_rank": range(1, 7), "ce_prob": [0.99, 0.95, 0.1, 0.7, 0.2, 0.1],
    })
    positives = pd.DataFrame({"term_id": ["t1"], "item_id": ["i0"]})
    triaged = label_and_triage(candidates, positives)
    assert not bool(triaged.loc[triaged.item_id.eq("i0"), "exclude_from_train"].iloc[0])
    assert bool(triaged.loc[triaged.item_id.eq("i1"), "exclude_from_train"].iloc[0])
    assert triaged.loc[triaged.item_id.eq("i3"), "sample_weight"].iloc[0] == 0.5
    assert add_query_views(triaged)["query_normalized"].iloc[0] == "ışık"


def test_negative_cap_is_per_term_and_deterministic():
    rows = pd.DataFrame({
        "term_id": ["t1"] * 6, "item_id": [f"i{i}" for i in range(6)], "label": [1, 1, 0, 0, 0, 0],
        "exclude_from_train": False,
    })
    sampled = sample_training_rows(rows, negatives_per_positive=1, seed=3)
    assert sampled.label.value_counts().to_dict() == {1: 2, 0: 2}
