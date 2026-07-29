import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apply_engine_residual import apply_positive_flips, selected_ids
from build_trendyol_domain_features import pair_text_features
from build_fast_structural_pseudolabels import add_fast_retrieval_score
from build_position_prior_variants import learn_rank_log_odds, rank_adjusted_probability
from trendyol_engine_replica import add_structural_residual_columns


class EngineReplicaTests(unittest.TestCase):
    def test_pair_text_features_keep_semantic_channels_separate(self):
        recall, jaccard, contained, taxonomy = pair_text_features(
            np.array([0, 1]), np.array([0, 1]),
            ["beyaz sweatshirt", "halı"],
            ["Beyaz kapüşonlu sweatshirt", "Modern kilim"],
            ["giyim/üst giyim", "ev/halı ve kilim"],
            ["marka", "marka"],
        )
        self.assertEqual(float(recall[0]), 1.0)
        self.assertGreater(float(jaccard[0]), 0.5)
        self.assertEqual(float(contained[0]), 0.0)
        self.assertEqual(float(taxonomy[1]), 1.0)

    def test_structural_residual_selects_exact_group_excess(self):
        rows = []
        for term_id, size in (("a", 102), ("b", 100)):
            for rank in range(1, size + 1):
                rows.append({
                    "id": f"{term_id}-{rank}", "term_id": term_id,
                    "item_id": f"i-{term_id}-{rank}", "candidate_count": size,
                    "engine_score": 1.0 - rank / 1000,
                    "replica_score": 1.0 - rank / 950,
                    "lexical_score": 1.0 - rank / 900,
                    "ty_cosine": 1.0 - rank / 800,
                    "ty_rank": rank,
                })
        scored = add_structural_residual_columns(pd.DataFrame(rows))
        residual = scored[scored["is_structural_residual"]]
        self.assertEqual(len(residual), 2)
        self.assertEqual(set(residual["id"]), {"a-101", "a-102"})
        self.assertTrue(residual["outside_agreement"].eq(3).all())

    def test_anchor_application_is_only_zero_to_one(self):
        anchor = pd.DataFrame({"id": ["a", "b", "c"], "prediction": [0, 1, 0]})
        residual = pd.DataFrame({
            "id": ["a", "b", "c"],
            "outside_agreement": [3, 3, 1],
            "forced_confidence": [2.0, 1.0, 3.0],
            "forced_margin": [1.0, 0.5, 2.0],
        })
        ids = selected_ids(residual, minimum_agreement=2)
        output, flips = apply_positive_flips(anchor, ids)
        self.assertEqual(flips, 1)
        self.assertEqual(output["prediction"].tolist(), [1, 1, 0])

    def test_fast_structural_score_selects_group_excess(self):
        rows = []
        for rank in range(102):
            rows.append({
                "id": f"a-{rank}", "term_id": "a", "item_id": f"i-{rank}",
                "overlap_title": float(102 - rank) / 102,
                "jaccard_title": 0.0, "overlap_category": 0.0, "brand_in_query": 0.0,
            })
        scored = add_fast_retrieval_score(pd.DataFrame(rows))
        residual = scored[scored["is_structural_residual"]]
        self.assertEqual(residual["id"].tolist(), ["a-100", "a-101"])

    def test_position_prior_preserves_forced_positive_tail(self):
        train = pd.DataFrame({
            "retrieval_rank": [1, 1, 100, 100], "label": [1, 1, 0, 0],
        })
        log_odds, _, _ = learn_rank_log_odds(train)
        adjusted = rank_adjusted_probability(
            np.array([0.5, 0.5, 0.1], dtype=np.float32),
            np.array([1, 100, 101], dtype=np.int32), log_odds, 1.0,
        )
        self.assertGreater(float(adjusted[0]), float(adjusted[1]))
        self.assertEqual(float(adjusted[2]), 1.0)


if __name__ == "__main__":
    unittest.main()
