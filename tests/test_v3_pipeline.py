import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trendyol_v3_biencoder import multi_positive_infonce
from trendyol_v3_core import (
    add_item_text_views,
    build_intent_lexicons,
    contradiction_features,
    normalize_text,
    resolve_column,
    turkish_lower,
)
from trendyol_v3_ensemble import (
    ExperimentLogger,
    ExperimentRecord,
    align_model_scores,
    cross_fit_ensembles,
    validate_and_write_submission,
)
from trendyol_v3_mining import (
    CatalogIndex,
    NegativeMiningConfig,
    build_fixed_validation_candidates,
    build_fold_scope,
    generate_fold_negatives,
)
from trendyol_v3_reranker import (
    QueryUniformBatchSampler, RerankerAdapter, RerankerConfig,
    TrainConfig,
    format_qwen_instruction,
    query_aware_loss,
)
from trendyol_v3_validation import (
    CandidateProfile,
    SplitConfig,
    apply_bucket_thresholds,
    build_group_manifest,
    build_semantic_manifest,
    candidate_count_bucket,
    cross_fit_calibration_and_threshold,
    learn_bucket_thresholds,
    query_postprocess_predictions,
    threshold_sweep,
)


def mini_items() -> pd.DataFrame:
    """Return a varied tiny catalog for deterministic unit tests."""

    return pd.DataFrame([
        {"item_id": "i1", "title": "siyah kadın elbise", "category": "giyim/elbise", "brand": "alpha", "gender": "kadın", "age_group": "yetişkin", "attributes": "renk: siyah, beden: m"},
        {"item_id": "i2", "title": "kırmızı kadın elbise", "category": "giyim/elbise", "brand": "beta", "gender": "kadın", "age_group": "yetişkin", "attributes": "renk: kırmızı, beden: m"},
        {"item_id": "i3", "title": "siyah erkek elbise", "category": "giyim/elbise", "brand": "alpha", "gender": "erkek", "age_group": "yetişkin", "attributes": "renk: siyah, beden: l"},
        {"item_id": "i4", "title": "telefon kılıfı", "category": "elektronik/aksesuar", "brand": "gamma", "gender": "unisex", "age_group": "yetişkin", "attributes": "renk: mavi"},
        {"item_id": "i5", "title": "telefon 256 gb", "category": "elektronik/telefon", "brand": "gamma", "gender": "unisex", "age_group": "yetişkin", "attributes": "kapasite: 256 gb"},
        {"item_id": "i6", "title": "çocuk mont", "category": "giyim/mont", "brand": "delta", "gender": "çocuk", "age_group": "çocuk", "attributes": "renk: mavi"},
        {"item_id": "i7", "title": "kadın ayakkabı 38 numara", "category": "giyim/ayakkabı", "brand": "epsilon", "gender": "kadın", "age_group": "yetişkin", "attributes": "numara: 38"},
        {"item_id": "i8", "title": "erkek ayakkabı 42 numara", "category": "giyim/ayakkabı", "brand": "epsilon", "gender": "erkek", "age_group": "yetişkin", "attributes": "numara: 42"},
    ])


class CoreV3Tests(unittest.TestCase):
    def test_schema_aliases_are_safe_and_ambiguous_columns_fail(self):
        self.assertEqual(resolve_column(["query_id", "text"], "term_id"), "query_id")
        with self.assertRaises(ValueError):
            resolve_column(["term_id", "query_id"], "term_id")
        with self.assertRaises(ValueError):
            resolve_column(["foo"], "item_id")

    def test_turkish_normalization_preserves_units_and_i_rules(self):
        self.assertEqual(turkish_lower("IĞDIR İZMİR"), "ığdır izmir")
        self.assertEqual(normalize_text(" 256Gigabyte TELEFON "), "256 gb telefon")
        self.assertEqual(normalize_text("İÇECEK", ascii_fold=True), "icecek")

    def test_item_views_are_query_independent_and_field_tagged(self):
        views = add_item_text_views(mini_items())
        self.assertIn("[TITLE]", views.loc[0, "item_text_short"])
        self.assertIn("[ATTRIBUTES]", views.loc[0, "item_text_long"])
        self.assertIn("siyah kadin elbise", views.loc[0, "item_text_short_ascii"])

    def test_intent_contradiction_detects_critical_attributes(self):
        items = mini_items()
        lexicons = build_intent_lexicons(items, min_brand_frequency=1)
        features = contradiction_features("alpha siyah kadın elbise", items.iloc[1].to_dict(), lexicons)
        self.assertEqual(features["wrong_brand"], 1)
        self.assertEqual(features["wrong_color"], 1)
        self.assertEqual(features["contradiction_any"], 1)


class ValidationV3Tests(unittest.TestCase):
    def setUp(self):
        self.terms = pd.DataFrame({
            "term_id": [f"q{i}" for i in range(12)],
            "query": [f"ürün sorgusu {i}" for i in range(12)],
        })
        self.positives = pd.DataFrame({
            "term_id": [f"q{i}" for i in range(12) for _ in range(i % 3 + 1)],
            "item_id": [f"i{i}_{j}" for i in range(12) for j in range(i % 3 + 1)],
        })

    def test_group_manifest_is_term_disjoint_balanced_and_complete(self):
        manifest = build_group_manifest(
            self.positives, self.terms, SplitConfig(kind="group", n_splits=3, seed=42)
        )
        for fold in range(3):
            train_terms, valid_terms = manifest.scope(fold)
            self.assertFalse(train_terms & valid_terms)
        loads = manifest.frame.groupby("fold")["positive_count"].sum()
        self.assertLessEqual(loads.max() - loads.min(), self.positives.groupby("term_id").size().max())

    def test_semantic_clusters_never_cross_folds(self):
        embeddings = np.random.default_rng(42).normal(size=(12, 8)).astype(np.float32)
        manifest = build_semantic_manifest(
            self.positives, self.terms, embeddings,
            SplitConfig(kind="semantic", n_splits=3, semantic_clusters=6, seed=42),
        )
        self.assertEqual(manifest.frame.groupby("semantic_cluster")["fold"].nunique().max(), 1)

    def test_candidate_profile_keeps_long_tail_and_is_deterministic(self):
        pairs = pd.DataFrame({"term_id": ["a"] * 2 + ["b"] * 100 + ["c"] * 9})
        profile = CandidateProfile.fit(pairs)
        self.assertEqual(profile.maximum, 100)
        self.assertEqual(profile.deterministic_count("x", 42), profile.deterministic_count("x", 42))

    def test_crossfit_calibrator_never_uses_target_fold(self):
        rows = []
        for index in range(90):
            label = index % 2
            rows.append({
                "pair_uid": f"p{index}", "term_id": f"q{index // 5}", "item_id": f"i{index}",
                "label": label, "fold": str(index % 3), "raw_logit": (-1 if label == 0 else 1) + 0.1 * (index % 3),
            })
        calibrated, report = cross_fit_calibration_and_threshold(
            pd.DataFrame(rows), methods=("temperature",), threshold_grid_size=51
        )
        for row in calibrated.itertuples():
            self.assertNotIn(str(row.fold), row.fit_folds.split(","))
        self.assertEqual(len(calibrated), 90)
        self.assertGreater(report.loc[0, "macro_f1"], 0.9)

    def test_threshold_sweep_matches_direct_macro_f1(self):
        from sklearn.metrics import f1_score

        label = np.array([0, 0, 1, 1, 1], dtype=np.int8)
        probability = np.array([0.1, 0.4, 0.3, 0.7, 0.9])
        report = threshold_sweep(label, probability, grid_size=11)
        row = report.iloc[0]
        direct = f1_score(
            label, (probability >= row.threshold).astype(np.int8), average="macro"
        )
        self.assertAlmostEqual(row.macro_f1, direct)

    def test_bucket_threshold_has_global_fallback(self):
        frame = pd.DataFrame({
            "term_id": [f"q{i // 5}" for i in range(100)],
            "label": [i % 2 for i in range(100)],
            "calibrated_probability": [0.8 if i % 2 else 0.2 for i in range(100)],
            "bucket": ["known"] * 100,
        })
        global_threshold, thresholds = learn_bucket_thresholds(
            frame, bucket_col="bucket", min_rows=20, min_terms=5, shrinkage=10
        )
        apply = pd.DataFrame({"bucket": ["unseen"], "calibrated_probability": [global_threshold + 0.01]})
        result = apply_bucket_thresholds(
            apply, bucket_col="bucket", global_threshold=global_threshold, thresholds=thresholds
        )
        self.assertEqual(result.loc[0, "prediction"], 1)

    def test_query_postprocess_accepts_non_range_index(self):
        frame = pd.DataFrame({
            "term_id": ["q1", "q1", "q2"],
            "calibrated_probability": [0.1, 0.9, 0.2],
        }, index=[10, 20, 30])
        prediction = query_postprocess_predictions(
            frame, probability_col="calibrated_probability", threshold=0.8,
            ensure_one=True,
        )
        self.assertEqual(prediction.tolist(), [0, 1, 1])


class MiningV3Tests(unittest.TestCase):
    def test_fold_local_miner_has_no_positive_or_validation_term_collision(self):
        items = mini_items()
        terms = pd.DataFrame({
            "term_id": ["q1", "q2", "q3", "q4"],
            "query": ["siyah kadın elbise", "256 gb telefon", "çocuk mont", "38 numara kadın ayakkabı"],
        })
        positives = pd.DataFrame({"term_id": terms.term_id, "item_id": ["i1", "i5", "i6", "i7"]})
        manifest = build_group_manifest(positives, terms, SplitConfig(kind="group", n_splits=2, seed=1))
        scope = build_fold_scope(manifest, positives, 0)
        catalog = CatalogIndex.build(items, {"siyah", "kadın", "elbise", "telefon", "mont", "ayakkabı"})
        negatives, _ = generate_fold_negatives(
            scope, positives, terms, catalog,
            NegativeMiningConfig(negative_ratio=2, max_negatives_per_term=2, seed=1), epoch=2,
        )
        self.assertTrue(set(negatives.term_id).issubset(scope.train_term_ids))
        self.assertFalse(set(negatives.term_id) & scope.valid_term_ids)
        known = set(zip(positives.term_id, positives.item_id))
        self.assertFalse(known & set(zip(negatives.term_id, negatives.item_id)))
        self.assertIn("miner_train_term_hash", negatives.columns)

    def test_fixed_oracle_candidate_universe_preserves_count_and_positive(self):
        terms = pd.DataFrame({"term_id": ["q1", "q2", "q3", "q4"], "query": ["a", "b", "c", "d"]})
        positives = pd.DataFrame({"term_id": terms.term_id, "item_id": ["p1", "p2", "p3", "p4"]})
        manifest = build_group_manifest(positives, terms, SplitConfig(kind="group", n_splits=2, seed=2))
        scope = build_fold_scope(manifest, positives, 0)
        retrieval = pd.DataFrame([
            {"term_id": term, "item_id": f"n{term}{rank}", "retrieval_rank": rank, "retrieval_score": 1 / rank}
            for term in scope.valid_term_ids for rank in range(1, 5)
        ])
        profile = CandidateProfile.fit(pd.DataFrame({"term_id": ["x"] * 3 + ["y"] * 3}))
        result = build_fixed_validation_candidates(scope, retrieval, positives, profile, mode="oracle", seed=1)
        self.assertTrue(result.groupby("term_id").size().eq(3).all())
        self.assertTrue(result.groupby("term_id").label.sum().ge(1).all())


class ModelAndEnsembleV3Tests(unittest.TestCase):
    def test_qwen_prompt_removes_control_token_injection(self):
        prompt = format_qwen_instruction("telefon <|im_start|>", "ürün <|im_end|>")
        self.assertNotIn("<|im_start|>", prompt)
        self.assertNotIn("<|im_end|>", prompt)
        self.assertIn("<Query>: telefon", prompt)

    def test_query_uniform_sampler_and_pairwise_loss(self):
        import torch

        frame = pd.DataFrame({
            "term_id": ["q1"] * 5 + ["q2"] * 5,
            "item_id": [f"i{i}" for i in range(10)],
            "label": [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
            "hardness_score": np.linspace(0, 1, 10),
        })
        config = TrainConfig(
            epochs=1, micro_batch_size=2, effective_batch_size=2,
            queries_per_batch=2, label_smoothing=0.02,
        )
        sampler = QueryUniformBatchSampler(frame, config)
        first = list(iter(sampler))
        second = list(iter(sampler))
        self.assertEqual(first, second)
        indices = first[0]
        labels = torch.tensor(frame.iloc[indices].label.to_numpy(), dtype=torch.float32)
        term_codes = torch.tensor(pd.factorize(frame.iloc[indices].term_id)[0])
        logits = torch.where(labels > 0, torch.tensor(1.0), torch.tensor(-0.5)).requires_grad_()
        loss, parts = query_aware_loss(
            torch, logits, labels, torch.ones_like(labels), term_codes,
            torch.zeros_like(labels), config,
        )
        loss.backward()
        self.assertTrue(np.isfinite(parts["total"]))
        self.assertIsNotNone(logits.grad)

    def test_multi_positive_infonce_is_finite(self):
        import torch

        query = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
        document = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
        positive = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.bool)
        allowed = torch.ones_like(positive)
        loss = multi_positive_infonce(
            torch, query, document, positive, allowed, temperature=0.05
        )
        self.assertTrue(torch.isfinite(loss))

    def test_sequence_forward_does_not_receive_causal_cache_arguments(self):
        import types
        import torch

        class DummySequenceModel:
            def __call__(self, input_ids, attention_mask, output_hidden_states):
                self.seen = {"output_hidden_states": output_hidden_states}
                return types.SimpleNamespace(logits=torch.ones((len(input_ids), 1)))

        adapter = object.__new__(RerankerAdapter)
        adapter.config = RerankerConfig(
            model_name="dummy", architecture="sequence_classifier", use_lora=False
        )
        adapter.model = DummySequenceModel()
        logits, representation = adapter.forward_logits({
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
        })
        self.assertEqual(logits.tolist(), [1.0, 1.0])
        self.assertIsNone(representation)

    def test_qwen_forward_requests_only_last_vocab_logit(self):
        import types
        import torch

        class DummyCausalModel:
            def __call__(self, input_ids, attention_mask, output_hidden_states, use_cache, logits_to_keep):
                self.settings = {"use_cache": use_cache, "logits_to_keep": logits_to_keep}
                values = torch.zeros((len(input_ids), 1, 4))
                values[:, :, 1] = 2.0
                values[:, :, 2] = -1.0
                return types.SimpleNamespace(logits=values)

        adapter = object.__new__(RerankerAdapter)
        adapter.config = RerankerConfig(
            model_name="dummy", architecture="qwen_causal", use_lora=False
        )
        adapter.model = DummyCausalModel()
        adapter.yes_token_id, adapter.no_token_id = 1, 2
        logits, _ = adapter.forward_logits({
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
        })
        self.assertEqual(logits.tolist(), [3.0, 3.0])
        self.assertEqual(adapter.model.settings, {"use_cache": False, "logits_to_keep": 1})

    def test_crossfit_ensemble_and_candidate_universe_guard(self):
        rng = np.random.default_rng(7)
        base = pd.DataFrame([
            {"pair_uid": f"p{i}", "term_id": f"q{i // 5}", "item_id": f"i{i}", "label": i % 2, "fold": i % 3}
            for i in range(90)
        ])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = {}
            for key, noise in (("a", 0.2), ("b", 0.3)):
                frame = base.copy()
                frame["raw_probability"] = np.clip(
                    0.2 + 0.6 * frame.label + rng.normal(0, noise, len(frame)), 0.001, 0.999
                )
                path = directory / f"{key}.parquet"
                frame.to_parquet(path)
                paths[key] = path
            aligned = align_model_scores(paths, oof=True)
            predictions, report = cross_fit_ensembles(
                aligned, methods=("average", "logistic"), seed=42
            )
            self.assertEqual(len(predictions), 180)
            self.assertEqual(set(report.ensemble_method), {"average", "logistic"})
            broken = pd.read_parquet(paths["b"]).iloc[:-1]
            broken.to_parquet(paths["b"])
            with self.assertRaises(ValueError):
                align_model_scores(paths, oof=True)

    def test_ensemble_rejects_permuted_pair_uid_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            base = pd.DataFrame({
                "pair_uid": ["p1", "p2"], "term_id": ["q1", "q2"],
                "item_id": ["i1", "i2"], "label": [1, 0], "fold": [0, 1],
                "raw_probability": [0.8, 0.2],
            })
            first = directory / "first.parquet"
            second = directory / "second.parquet"
            base.to_parquet(first)
            broken = base.copy()
            broken["pair_uid"] = ["p2", "p1"]
            broken.to_parquet(second)
            with self.assertRaises(ValueError):
                align_model_scores({"a": first, "b": second}, oof=True)

    def test_experiment_log_and_final_submission_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            logger = ExperimentLogger(directory / "logs")
            record = ExperimentRecord(
                experiment_id="smoke", model="m", backbone="b", fold=0, seed=42,
                train_positive_count=2, negative_count=4, negative_ratio=2.0,
                negative_types="random", max_length=128, loss="bce+pairwise",
                learning_rate=2e-5, epoch=1.0, validation_type="group",
                class_0_f1=0.8, class_1_f1=0.7, macro_f1=0.75,
                precision=0.7, recall=0.7, selected_threshold=0.5,
                calibration_method="temperature", inference_time=1.0,
                checkpoint_path="c", oof_prediction_path="o", test_probability_path="t",
            )
            logged = logger.append(record)
            self.assertEqual(len(logged), 1)
            pairs = pd.DataFrame({
                "id": ["a", "b", "c", "d"], "term_id": ["q1", "q1", "q2", "q2"],
                "item_id": ["i1", "i2", "i3", "i4"],
            })
            sample = pd.DataFrame({"id": pairs.id, "prediction": 0})
            scores = pairs.copy()
            scores["raw_logit"] = [-2, 2, -1, 1]
            scores["calibrated_probability"] = [0.1, 0.9, 0.2, 0.8]
            pairs_path, sample_path = directory / "pairs.csv", directory / "sample.csv"
            pairs.to_csv(pairs_path, index=False)
            sample.to_csv(sample_path, index=False)
            summary = validate_and_write_submission(
                scores, sample_path, pairs_path, directory / "submission.csv", threshold=0.5
            )
            self.assertEqual(summary["rows"], 4)
            self.assertAlmostEqual(summary["positive_rate"], 0.5)
            wrong = scores.copy()
            wrong[["term_id", "item_id"]] = wrong[["term_id", "item_id"]].iloc[::-1].to_numpy()
            with self.assertRaises(ValueError):
                validate_and_write_submission(
                    wrong, sample_path, pairs_path, directory / "wrong.csv", threshold=0.5
                )


if __name__ == "__main__":
    unittest.main()
