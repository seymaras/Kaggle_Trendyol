import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_features import add_query_level_features
from experiment_utils import stable_term_folds, validate_probability_frame, validate_submission
from make_ratio_submissions import prediction_at_ratio
from text_features import (
    build_lexical_features, category_parts, normalize_category_path,
    parse_attributes, select_attributes,
)
from trendyol_query_hard_negative_mining import choose_negatives, target_negative_count


class TextFeatureTests(unittest.TestCase):
    def test_category_path_is_preserved_and_split(self):
        raw = "Elektronik / Telefon / Cep Telefonu"
        self.assertEqual(normalize_category_path(raw), "elektronik/telefon/cep telefonu")
        self.assertEqual(category_parts(raw)["category_depth"], 3)
        self.assertEqual(category_parts(raw)["leaf_category"], "cep telefonu")

    def test_attribute_parser_and_selection(self):
        raw = "renk: Siyah, kapasite: 128 GB, menşei: TR"
        parsed = parse_attributes(raw)
        self.assertEqual(parsed["renk"], "siyah")
        self.assertIn("kapasite: 128 gb", select_attributes(raw, "128 gb telefon"))

    def test_lexical_features_include_numeric_and_category_signals(self):
        frame = pd.DataFrame([{
            "query": "siyah s23 128 gb telefon", "title": "S23 telefon 128 GB siyah",
            "category": "elektronik/telefon/cep telefonu", "brand": "samsung",
            "gender": "", "age_group": "", "attributes": "renk: siyah, kapasite: 128 gb",
        }])
        features = build_lexical_features(frame)
        self.assertEqual(features.loc[0, "category_depth"], 3)
        self.assertEqual(features.loc[0, "number_match"], 1)
        self.assertEqual(features.loc[0, "model_code_match"], 1)


class PipelineSafetyTests(unittest.TestCase):
    def test_stable_folds(self):
        terms = pd.Series(["a", "b", "a", "c"])
        first = stable_term_folds(terms, 5, 42)
        second = stable_term_folds(terms, 5, 42)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.term_id.nunique(), 3)

    def test_ratio_is_exact_and_deterministic_with_ties(self):
        probability = np.array([0.5] * 10)
        ids = np.array([f"id-{i}" for i in range(10)])
        first = prediction_at_ratio(probability, ids, 0.2)
        second = prediction_at_ratio(probability, ids, 0.2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.sum(), 2)

    def test_submission_validation(self):
        sample = pd.DataFrame({"id": ["a", "b"], "prediction": [0, 0]})
        submission = pd.DataFrame({"id": ["a", "b"], "prediction": [1, 0]})
        validate_submission(submission, sample)
        probability = pd.DataFrame({
            "id": ["a", "b"], "term_id": ["q", "q"], "item_id": ["x", "y"],
            "probability": [0.1, 0.9],
        })
        validate_probability_frame(probability, 2)

    def test_query_features(self):
        frame = pd.DataFrame({
            "term_id": ["q", "q"], "item_id": ["x", "y"],
            "semantic_score": [0.8, 0.2], "top_category": ["a", "a"],
            "leaf_category": ["b", "c"],
        })
        result = add_query_level_features(frame)
        self.assertEqual(result.loc[0, "score_rank"], 1)
        self.assertAlmostEqual(result.loc[1, "score_gap_to_top"], 0.6)
        self.assertEqual(result.loc[0, "candidate_top_category_frequency"], 1.0)

    def test_negative_buckets_exclude_known_positive(self):
        candidates = pd.DataFrame({
            "term_id": ["q"] * 60, "query": ["telefon"] * 60,
            "item_id": [f"i{i}" for i in range(60)], "retrieval_rank": range(1, 61),
            "retrieval_score": np.linspace(1, 0, 60), "retrieval_source": ["word"] * 60,
            "item_position": range(60),
        })
        selected, easy_count = choose_negatives(candidates, {"i0", "i10"})
        self.assertFalse(set(selected.item_id) & {"i0", "i10"})
        self.assertLessEqual(len(selected) + easy_count, 12)

    def test_negative_target_scales_with_positive_count(self):
        self.assertEqual(target_negative_count(1), 12)
        self.assertEqual(target_negative_count(10), 20)
        self.assertEqual(target_negative_count(100), 40)


if __name__ == "__main__":
    unittest.main()
