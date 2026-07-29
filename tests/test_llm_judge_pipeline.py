from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.build_llm_judge_pool import count_preserving_top_k, exact_top_k
from src.run_llm_judge_colab import finalize, parse_gpt_oss_final_label


def test_parse_gpt_oss_final_label_reads_only_harmony_final_channel() -> None:
    prefix = (
        "<|start|>assistant<|channel|>analysis<|message|>"
        "The analysis mentions both 0 and 1.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
    )
    assert parse_gpt_oss_final_label(prefix + "0<|return|>") == 0
    assert parse_gpt_oss_final_label(prefix + "1.<|end|>") == 1


def test_parse_gpt_oss_final_label_rejects_missing_or_verbose_final() -> None:
    for completion in (
        "1",
        "<|channel|>analysis<|message|>1<|end|>",
        "<|channel|>final<|message|>label=1<|return|>",
    ):
        try:
            parse_gpt_oss_final_label(completion)
        except ValueError:
            pass
        else:
            raise AssertionError(f"completion should be rejected: {completion!r}")


def test_count_preserving_top_k_keeps_each_group_count() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.9], dtype=np.float32)
    term_codes = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    anchor = np.array([0, 0, 1, 1, 0, 1], dtype=np.uint8)

    prediction, boundary = count_preserving_top_k(scores, term_codes, anchor)

    assert prediction.tolist() == [1, 1, 0, 0, 0, 1]
    assert np.bincount(term_codes, weights=prediction).tolist() == [2.0, 1.0]
    assert np.isfinite(boundary).all()


def test_exact_top_k_has_exact_cardinality() -> None:
    prediction, boundary = exact_top_k(np.array([0.1, 0.4, 0.2, 0.3]), 2)
    assert prediction.tolist() == [0, 1, 0, 1]
    assert prediction.sum() == 2
    assert np.isclose(boundary, 0.3)


def test_finalize_writes_valid_free_and_count_preserving_candidates(tmp_path) -> None:
    input_dir = tmp_path / "input"
    work_dir = tmp_path / "work"
    input_dir.mkdir()

    anchor = pd.DataFrame(
        {
            "id": [f"id_{index}" for index in range(8)],
            "prediction": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    pool = pd.DataFrame(
        {
            "id": ["id_0", "id_1", "id_2", "id_3", "id_4", "id_5"],
            "term_id": ["a", "a", "a", "a", "b", "b"],
            "row_position": [0, 1, 2, 3, 4, 5],
            "anchor_prediction": [0, 0, 1, 1, 0, 1],
            "alternative_prediction": [1, 1, 0, 0, 1, 0],
            "source_votes": [2, 1, 2, 1, 2, 2],
            "priority": [21.0, 11.0, 22.0, 12.0, 20.0, 20.0],
        }
    )
    anchor.to_parquet(input_dir / "anchor_v6.parquet", index=False)
    pool.to_parquet(input_dir / "llm_judge_pool.parquet", index=False)
    (input_dir / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "anchor_v6.parquet": {},
                    "llm_judge_pool.parquet": {},
                }
            }
        ),
        encoding="utf-8",
    )

    confidence = np.array([0.90, 0.70, 0.85, 0.70, 0.95, 0.95], dtype=np.float32)
    labels = pool["alternative_prediction"].to_numpy(dtype=np.int8)
    p_relevant = np.where(labels == 1, confidence, 1.0 - confidence)
    for slug in ("model_a", "model_b"):
        vote_dir = work_dir / "votes" / slug
        vote_dir.mkdir(parents=True)
        run_signature = f"test-signature-{slug}"
        pd.DataFrame(
            {
                "id": pool["id"],
                "label": labels,
                "p_relevant": p_relevant,
                "confidence": confidence,
                "run_signature": run_signature,
            }
        ).to_parquet(vote_dir / "part_00000.parquet", index=False)
        (vote_dir / "model_run.json").write_text(
            json.dumps({"run_signature": run_signature}), encoding="utf-8"
        )

    args = SimpleNamespace(
        input_dir=input_dir,
        work_dir=work_dir,
        strict_threshold=0.80,
        medium_threshold=0.65,
        second_model_prefilter=0.55,
    )
    finalize(args, [("model_a", "repo/a"), ("model_b", "repo/b")])

    output_dir = work_dir / "output"
    expected = {
        "llm_consensus_strict.csv",
        "llm_consensus_medium.csv",
        "llm_consensus_broad.csv",
        "llm_count_preserving_strict.csv",
        "llm_count_preserving_medium.csv",
    }
    assert {path.name for path in output_dir.glob("*.csv")} == expected
    for path in output_dir.glob("*.csv"):
        candidate = pd.read_csv(path)
        assert candidate["id"].tolist() == anchor["id"].tolist()
        assert set(candidate["prediction"].unique()) <= {0, 1}
        assert len(candidate) == len(anchor)

    report = json.loads((output_dir / "llm_judge_report.json").read_text())
    assert report["kaggle_submission_called"] is False
    assert report["candidates"]["llm_count_preserving_strict"]["flips"] == 4


def test_finalize_three_model_cascade_adds_vetoed_candidates(tmp_path) -> None:
    input_dir = tmp_path / "input"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    anchor = pd.DataFrame(
        {"id": [f"id_{i}" for i in range(6)], "prediction": [0, 0, 1, 1, 0, 1]}
    )
    pool = pd.DataFrame(
        {
            "id": anchor["id"],
            "term_id": ["a", "a", "a", "a", "b", "b"],
            "row_position": np.arange(6),
            "anchor_prediction": anchor["prediction"],
            "alternative_prediction": 1 - anchor["prediction"],
            "source_votes": [3, 2, 3, 2, 3, 3],
            "priority": [30.0, 20.0, 30.0, 20.0, 30.0, 30.0],
        }
    )
    anchor.to_parquet(input_dir / "anchor_v6.parquet", index=False)
    pool.to_parquet(input_dir / "llm_judge_pool.parquet", index=False)
    (input_dir / "manifest.json").write_text(
        json.dumps({"files": {"anchor_v6.parquet": {}, "llm_judge_pool.parquet": {}}})
    )

    alternative = pool["alternative_prediction"].to_numpy(dtype=np.int8)
    confidence = np.full(6, 0.90, dtype=np.float32)
    for slug, ids, labels in (
        ("q", pool["id"], alternative),
        ("m", pool["id"], alternative),
        ("g", pool["id"], np.array([1, 1, 0, 1, 1, 0], dtype=np.int8)),
    ):
        vote_dir = work_dir / "votes" / slug
        vote_dir.mkdir(parents=True)
        signature = f"sig-{slug}"
        local_confidence = confidence[: len(ids)]
        pd.DataFrame(
            {
                "id": list(ids),
                "label": labels,
                "p_relevant": np.where(labels == 1, local_confidence, 1 - local_confidence),
                "confidence": local_confidence,
                "run_signature": signature,
            }
        ).to_parquet(vote_dir / "part_00000.parquet", index=False)
        (vote_dir / "model_run.json").write_text(json.dumps({"run_signature": signature}))

    args = SimpleNamespace(
        input_dir=input_dir,
        work_dir=work_dir,
        strict_threshold=0.80,
        medium_threshold=0.65,
        second_model_prefilter=0.55,
        third_model_prefilter=0.65,
    )
    finalize(args, [("q", "repo/q"), ("m", "repo/m"), ("g", "repo/g")])
    report = json.loads((work_dir / "output/llm_judge_report.json").read_text())
    assert report["model_rows_judged"] == [6, 6, 6]
    assert report["both_models_choose_alternative"] == 6
    assert report["all_three_choose_alternative"] == 5
    assert report["candidates"]["llm_qwen_mistral_medium"]["flips"] == 6
    assert report["candidates"]["llm_triple_consensus_medium"]["flips"] == 5


if __name__ == "__main__":
    test_count_preserving_top_k_keeps_each_group_count()
    test_exact_top_k_has_exact_cardinality()
    with tempfile.TemporaryDirectory(prefix="llm_judge_test_") as temporary:
        test_finalize_writes_valid_free_and_count_preserving_candidates(
            Path(temporary)
        )
    with tempfile.TemporaryDirectory(prefix="llm_cascade_test_") as temporary:
        test_finalize_three_model_cascade_adds_vetoed_candidates(Path(temporary))
    print("llm judge pipeline tests: OK")
