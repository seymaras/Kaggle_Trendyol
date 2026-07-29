#!/usr/bin/env python3
"""Colab'da guclu ve yeniden baslatilabilir embedding feature uretimi.

Bu dosya tek Colab hucresine yapistirilarak veya `%run` ile calistirilabilir.
Colab Pro+ A100 uzerinde Qwen3-Embedding-8B modelinin tam 4096 boyutlu
embedding'lerini Google Drive'a parca parca yazar ve train/submission
ciftleri icin normalize cosine benzerligi uretir.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Kullanici ayarlari
# ---------------------------------------------------------------------------
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/lady_recall_embeddings")
DATA_DIR = Path("/content/data")

KAGGLE_INPUT_DATASET = "seymaaras/lady-recall-embeddings-input"
KAGGLE_OUTPUT_DATASET = "seymaaras/lady-recall-embeddings"
KAGGLE_OUTPUT_TITLE = "lady-recall-embeddings"

# Colab Pro+ A100 icin en guclu Qwen3 embedding modeli ve tam embedding boyutu.
# Gerekirse ortam degiskenleriyle daha kucuk model/boyut secilebilir.
MODEL_SELECTION = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "512"))
TRUNCATE_DIM = int(os.getenv("EMBED_DIM", "4096"))
BATCH_SIZE_OVERRIDE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

EMBED_WRITE_CHUNK = 10_000
PAIR_COSINE_CHUNK = 10_000
EMBED_STORAGE_DTYPE = "float16"
UPLOAD_TO_KAGGLE = True

# Qwen model karti, sorgu tarafinda Ingilizce ve goreve ozel instruct oneriyor.
QUERY_PROMPT = (
    "Instruct: Given a Turkish e-commerce search query, retrieve product "
    "descriptions that are relevant and satisfy all requested attributes.\n"
    "Query: "
)

REQUIRED_INPUT_FILES = (
    "items_enriched.parquet",
    "terms.parquet",
    "train_with_negatives.parquet",
    "submission_pairs.csv",
)


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    """Komutu yazdirir, calistirir ve basarisizligi sessizce yutmaz."""
    print("$", " ".join(command), flush=True)
    return subprocess.run(
        command,
        check=not capture,
        capture_output=capture,
        text=capture,
    )


def install_dependencies() -> None:
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "kaggle>=1.7,<2",
        "sentence-transformers>=5.1,<6",
        "transformers>=4.51,<5",
    ])


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Bu script Google Colab icin hazirlandi.") from exc
    drive.mount("/content/drive", force_remount=False)
    DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_kaggle_credentials() -> None:
    credential_path = Path("/root/.kaggle/kaggle.json")
    env_ready = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    if credential_path.exists() or env_ready:
        return

    from google.colab import files

    print("Kaggle API anahtarinizi (kaggle.json) yukleyin:", flush=True)
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("kaggle.json yuklenmedi.")

    name, payload = next(iter(uploaded.items()))
    if not name.lower().endswith(".json"):
        raise ValueError(f"JSON dosyasi bekleniyordu, gelen: {name}")
    try:
        parsed = json.loads(payload.decode("utf-8"))
        if not parsed.get("username") or not parsed.get("key"):
            raise ValueError
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Yuklenen dosya gecerli bir kaggle.json degil.") from exc

    credential_path.parent.mkdir(parents=True, exist_ok=True)
    credential_path.write_bytes(payload)
    credential_path.chmod(0o600)


def ensure_input_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED_INPUT_FILES if not (DATA_DIR / name).exists()]
    if missing:
        print("Eksik girdiler indiriliyor:", ", ".join(missing), flush=True)
        run([
            "kaggle",
            "datasets",
            "download",
            KAGGLE_INPUT_DATASET,
            "-p",
            str(DATA_DIR),
            "--unzip",
        ])

    missing = [name for name in REQUIRED_INPUT_FILES if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Dataset icinde eksik dosyalar var: {missing}")


def clean_text(series):
    cleaned = series.fillna("").astype("string").str.strip()
    return cleaned.mask(cleaned.eq(""), " ")


def normalize_ids(series, name: str):
    """CSV ve Parquet ID'lerini ayni string tipine getirir."""
    import pandas as pd

    if pd.api.types.is_float_dtype(series.dtype):
        non_null = series.dropna()
        if len(non_null) and (non_null % 1 == 0).all():
            normalized = series.round().astype("Int64").astype("string")
        else:
            normalized = series.astype("string")
    else:
        normalized = series.astype("string")
    normalized = normalized.str.strip()
    bad = normalized.isna() | normalized.eq("")
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum()):,} bos/gecersiz ID var.")
    return normalized


def validate_unique(frame, id_column: str, label: str) -> None:
    duplicates = frame[id_column].duplicated(keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, id_column].head(5).tolist()
        raise ValueError(
            f"{label}: {int(duplicates.sum()):,} tekrarli {id_column} var. "
            f"Ornek: {examples}"
        )


def frame_fingerprint(frame, columns: list[str]) -> str:
    """Cache'in yalnizca ayni ID/metin sirasi icin kullanilmasini saglar."""
    import pandas as pd

    digest = hashlib.sha256()
    digest.update(str(len(frame)).encode("utf-8"))
    digest.update("\x1f".join(columns).encode("utf-8"))
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    digest.update(values.tobytes())
    return digest.hexdigest()


def load_catalogs():
    import pandas as pd

    items = pd.read_parquet(
        DATA_DIR / "items_enriched.parquet",
        columns=["item_id", "item_text"],
    )
    terms = pd.read_parquet(DATA_DIR / "terms.parquet", columns=["term_id", "query"])

    items["item_id"] = normalize_ids(items["item_id"], "items.item_id")
    terms["term_id"] = normalize_ids(terms["term_id"], "terms.term_id")
    items["item_text"] = clean_text(items["item_text"])
    terms["query"] = clean_text(terms["query"])
    validate_unique(items, "item_id", "items")
    validate_unique(terms, "term_id", "terms")

    print(f"Katalog: {len(items):,} urun | {len(terms):,} sorgu", flush=True)
    return items, terms


def choose_model(torch_module) -> tuple[str, int, float]:
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU bulunamadi. Colab > Runtime > Change runtime type > GPU secin."
        )

    total_gb = torch_module.cuda.get_device_properties(0).total_memory / (1024**3)
    if MODEL_SELECTION != "auto":
        model_name = MODEL_SELECTION
        batch_size = 8
    elif total_gb >= 30:
        model_name, batch_size = "Qwen/Qwen3-Embedding-8B", 8
    elif total_gb >= 14:
        model_name, batch_size = "Qwen/Qwen3-Embedding-4B", 4
    else:
        model_name, batch_size = "Qwen/Qwen3-Embedding-0.6B", 32

    if BATCH_SIZE_OVERRIDE > 0:
        batch_size = BATCH_SIZE_OVERRIDE
    return model_name, batch_size, total_gb


class AdaptiveEncoder:
    """CUDA OOM olursa batch'i yariya indirip ayni isi yeniden dener."""

    def __init__(self, model, torch_module, batch_size: int):
        self.model = model
        self.torch = torch_module
        self.batch_size = batch_size

    def encode(self, texts: list[str], role: str):
        kwargs = {}
        if role == "query":
            kwargs["prompt"] = QUERY_PROMPT
        elif role != "document":
            raise ValueError(f"Bilinmeyen embedding rolu: {role}")

        while True:
            try:
                return self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    truncate_dim=TRUNCATE_DIM,
                    **kwargs,
                )
            except self.torch.cuda.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                gc.collect()
                self.torch.cuda.empty_cache()
                print(
                    f"CUDA bellegi yetmedi; batch_size={self.batch_size} ile yeniden deneniyor.",
                    flush=True,
                )


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    model_name, batch_size, gpu_gb = choose_model(torch)
    device_name = torch.cuda.get_device_name(0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True

    print(
        f"Model: {model_name} | GPU: {device_name} ({gpu_gb:.1f} GB) | "
        f"dtype={dtype} | batch={batch_size}",
        flush=True,
    )
    model = SentenceTransformer(
        model_name,
        device="cuda",
        model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
        tokenizer_kwargs={"padding_side": "left"},
        truncate_dim=TRUNCATE_DIM,
    )
    model.max_seq_length = MAX_LENGTH
    model.eval()
    return model_name, AdaptiveEncoder(model, torch, batch_size)


def smoke_test(encoder: AdaptiveEncoder) -> int:
    import numpy as np

    queries = ["kirmizi kadin elbise", "bluetooth kulaklik"]
    documents = [
        "kirmizi renk kadin elbise, midi boy",
        "kablosuz bluetooth kulaklik ve sarj kutusu",
    ]
    query_vectors = encoder.encode(queries, "query")
    document_vectors = encoder.encode(documents, "document")
    scores = query_vectors @ document_vectors.T
    if not np.isfinite(scores).all() or scores.shape != (2, 2):
        raise RuntimeError("Embedding duman testi gecersiz sonuc uretti.")
    if not (scores[0, 0] > scores[0, 1] and scores[1, 1] > scores[1, 0]):
        raise RuntimeError(f"Embedding duman testi anlamsal kontrolu gecemedi: {scores}")
    print(f"Duman testi gecti | boyut={query_vectors.shape[1]} | skorlar={scores.round(3)}")
    return int(query_vectors.shape[1])


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def build_embedding_cache(
    *,
    encoder: AdaptiveEncoder,
    texts,
    role: str,
    model_name: str,
    dimension: int,
    fingerprint: str,
    output_path: Path,
):
    import numpy as np

    metadata_path = output_path.with_suffix(".meta.json")
    progress_path = output_path.with_suffix(".progress.json")
    spec = {
        "model_name": model_name,
        "role": role,
        "prompt": QUERY_PROMPT if role == "query" else None,
        "max_length": MAX_LENGTH,
        "dimension": dimension,
        "storage_dtype": EMBED_STORAGE_DTYPE,
        "row_count": len(texts),
        "source_fingerprint": fingerprint,
    }

    metadata = read_json(metadata_path)
    if metadata and metadata.get("status") == "complete" and metadata.get("spec") == spec:
        try:
            cached = np.load(output_path, mmap_mode="r")
            if cached.shape == (len(texts), dimension) and str(cached.dtype) == EMBED_STORAGE_DTYPE:
                print(f"Cache kullaniliyor: {output_path.name} {cached.shape}", flush=True)
                return cached
        except (FileNotFoundError, ValueError, OSError):
            pass

    progress = read_json(progress_path)
    can_resume = (
        output_path.exists()
        and progress is not None
        and progress.get("spec") == spec
        and 0 <= int(progress.get("next_row", -1)) <= len(texts)
    )
    if can_resume:
        vectors = np.lib.format.open_memmap(output_path, mode="r+")
        start_row = int(progress["next_row"])
        if vectors.shape != (len(texts), dimension) or str(vectors.dtype) != EMBED_STORAGE_DTYPE:
            can_resume = False

    if not can_resume:
        vectors = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=EMBED_STORAGE_DTYPE,
            shape=(len(texts), dimension),
        )
        start_row = 0
        atomic_write_json(progress_path, {"spec": spec, "next_row": 0})

    if start_row:
        print(f"Cache devam ediyor: {output_path.name} satir={start_row:,}", flush=True)
    started = time.time()
    for start in range(start_row, len(texts), EMBED_WRITE_CHUNK):
        end = min(start + EMBED_WRITE_CHUNK, len(texts))
        batch_vectors = encoder.encode(texts.iloc[start:end].tolist(), role)
        if batch_vectors.shape != (end - start, dimension):
            raise RuntimeError(
                f"Beklenmeyen embedding sekli: {batch_vectors.shape}; "
                f"beklenen={(end - start, dimension)}"
            )
        if not np.isfinite(batch_vectors).all():
            raise RuntimeError(f"Embedding icinde NaN/Inf var: satir {start:,}:{end:,}")
        vectors[start:end] = batch_vectors.astype(EMBED_STORAGE_DTYPE)
        vectors.flush()
        atomic_write_json(progress_path, {"spec": spec, "next_row": end})
        print(
            f"  {role}: {end:,}/{len(texts):,} ({time.time() - started:.0f}s)",
            flush=True,
        )

    del vectors
    metadata = {
        "status": "complete",
        "spec": spec,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(metadata_path, metadata)
    progress_path.unlink(missing_ok=True)
    return np.load(output_path, mmap_mode="r")


def parquet_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema.names)


def load_pairs(path: Path):
    import pandas as pd

    wanted = ["id", "term_id", "item_id", "label"]
    if path.suffix.lower() == ".parquet":
        available = parquet_columns(path)
        pairs = pd.read_parquet(path, columns=[name for name in wanted if name in available])
    else:
        pairs = pd.read_csv(
            path,
            usecols=lambda name: name in wanted,
            dtype={"id": "string", "term_id": "string", "item_id": "string"},
        )
    required = {"term_id", "item_id"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"{path.name}: eksik kolonlar: {sorted(missing)}")

    pairs["term_id"] = normalize_ids(pairs["term_id"], f"{path.name}.term_id")
    pairs["item_id"] = normalize_ids(pairs["item_id"], f"{path.name}.item_id")
    if "id" in pairs:
        # Join uyumlulugu icin Parquet'teki orijinal id tipini koru.
        bad_id = pairs["id"].isna() | pairs["id"].astype("string").str.strip().eq("")
        if bad_id.any():
            raise ValueError(f"{path.name}.id: {int(bad_id.sum()):,} bos/gecersiz ID var.")
    return pairs


def cosine_features(
    *,
    pairs,
    item_index,
    term_index,
    item_vectors,
    term_vectors,
    tag: str,
):
    import numpy as np

    item_positions = item_index.get_indexer(pairs["item_id"])
    term_positions = term_index.get_indexer(pairs["term_id"])
    bad = (item_positions < 0) | (term_positions < 0)
    if bad.any():
        examples = pairs.loc[bad, ["term_id", "item_id"]].head(8).to_dict("records")
        raise ValueError(
            f"{tag}: katalogla eslesmeyen {int(bad.sum()):,}/{len(pairs):,} cift var. "
            f"Ornek: {examples}"
        )

    scores = np.empty(len(pairs), dtype=np.float32)
    started = time.time()
    for start in range(0, len(pairs), PAIR_COSINE_CHUNK):
        end = min(start + PAIR_COSINE_CHUNK, len(pairs))
        queries = np.asarray(term_vectors[term_positions[start:end]], dtype=np.float32)
        documents = np.asarray(item_vectors[item_positions[start:end]], dtype=np.float32)

        # float16 Drive cache'i nedeniyle normlari tekrar duzeltir.
        numerator = np.einsum("ij,ij->i", queries, documents, optimize=True)
        denominator = np.linalg.norm(queries, axis=1) * np.linalg.norm(documents, axis=1)
        scores[start:end] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        if start == 0 or end == len(pairs) or (start // PAIR_COSINE_CHUNK) % 10 == 0:
            print(f"  {tag}: {end:,}/{len(pairs):,} ({time.time() - started:.0f}s)", flush=True)

    if not np.isfinite(scores).all():
        raise RuntimeError(f"{tag}: cosine sonucunda NaN/Inf olustu.")
    return scores


def atomic_write_parquet(frame, output_path: Path) -> None:
    temp_path = output_path.with_name(output_path.stem + ".tmp.parquet")
    frame.to_parquet(temp_path, index=False)
    os.replace(temp_path, output_path)


def build_pair_output(
    *,
    pair_path: Path,
    output_path: Path,
    item_index,
    term_index,
    item_vectors,
    term_vectors,
    tag: str,
) -> dict:
    import numpy as np
    import pandas as pd

    pairs = load_pairs(pair_path)
    if pairs.empty:
        raise ValueError(f"{pair_path.name}: cift dosyasi bos.")
    print(f"{tag} ciftleri: {len(pairs):,}", flush=True)
    scores = cosine_features(
        pairs=pairs,
        item_index=item_index,
        term_index=term_index,
        item_vectors=item_vectors,
        term_vectors=term_vectors,
        tag=tag,
    )

    # Eski cikti semasini koru; id yoksa guvenli join icin cifti anahtar yap.
    key_columns = ["id"] if "id" in pairs else ["term_id", "item_id"]
    output = pairs[key_columns].copy()
    output["embedding_cosine"] = scores
    atomic_write_parquet(output, output_path)

    summary = {
        "rows": len(output),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }
    if "label" in pairs:
        by_label = pd.DataFrame({"label": pairs["label"], "score": scores}).groupby("label")["score"].agg(
            ["count", "mean", "std"]
        )
        print(f"{tag} label kontrolu:\n{by_label}", flush=True)
        summary["by_label"] = {
            str(index): {key: float(value) for key, value in row.items()}
            for index, row in by_label.to_dict("index").items()
        }
    print(f"Kaydedildi: {output_path} | ort={summary['mean']:.4f}", flush=True)
    return summary


def upload_outputs(output_files: list[Path], run_metadata_path: Path, model_name: str) -> None:
    if not UPLOAD_TO_KAGGLE:
        print("Kaggle upload kapali.")
        return

    upload_dir = Path("/content/upload_embeddings_v2")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    for path in [*output_files, run_metadata_path]:
        shutil.copy2(path, upload_dir / path.name)

    atomic_write_json(
        upload_dir / "dataset-metadata.json",
        {
            "title": KAGGLE_OUTPUT_TITLE,
            "id": KAGGLE_OUTPUT_DATASET,
            "licenses": [{"name": "CC0-1.0"}],
        },
    )
    created = run([
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(upload_dir),
        "-r",
        "zip",
        "--dir-mode",
        "zip",
    ], capture=True)
    combined = f"{created.stdout}\n{created.stderr}".strip()
    if created.returncode == 0:
        print(combined, flush=True)
        return
    if "already exists" not in combined.lower():
        raise RuntimeError(f"Kaggle dataset create basarisiz:\n{combined}")

    message = f"Qwen3 embedding cosine: {model_name.split('/')[-1]}"
    run([
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(upload_dir),
        "-m",
        message,
        "-r",
        "zip",
        "--dir-mode",
        "zip",
    ])


def main() -> None:
    started = time.time()
    mount_drive()
    install_dependencies()
    ensure_kaggle_credentials()
    ensure_input_data()

    import pandas as pd

    items, terms = load_catalogs()
    item_fingerprint = frame_fingerprint(items, ["item_id", "item_text"])
    term_fingerprint = frame_fingerprint(terms, ["term_id", "query"])

    model_name, encoder = load_model()
    dimension = smoke_test(encoder)
    model_slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    cache_prefix = f"{model_slug}_d{dimension}_l{MAX_LENGTH}"

    item_vectors = build_embedding_cache(
        encoder=encoder,
        texts=items["item_text"],
        role="document",
        model_name=model_name,
        dimension=dimension,
        fingerprint=item_fingerprint,
        output_path=DRIVE_OUTPUT_DIR / f"{cache_prefix}_items.npy",
    )
    term_vectors = build_embedding_cache(
        encoder=encoder,
        texts=terms["query"],
        role="query",
        model_name=model_name,
        dimension=dimension,
        fingerprint=term_fingerprint,
        output_path=DRIVE_OUTPUT_DIR / f"{cache_prefix}_terms.npy",
    )

    item_index = pd.Index(items["item_id"])
    term_index = pd.Index(terms["term_id"])
    train_output = DRIVE_OUTPUT_DIR / "embedding_features_train.parquet"
    submission_output = DRIVE_OUTPUT_DIR / "embedding_features_submission.parquet"

    train_summary = build_pair_output(
        pair_path=DATA_DIR / "train_with_negatives.parquet",
        output_path=train_output,
        item_index=item_index,
        term_index=term_index,
        item_vectors=item_vectors,
        term_vectors=term_vectors,
        tag="train",
    )
    submission_summary = build_pair_output(
        pair_path=DATA_DIR / "submission_pairs.csv",
        output_path=submission_output,
        item_index=item_index,
        term_index=term_index,
        item_vectors=item_vectors,
        term_vectors=term_vectors,
        tag="submission",
    )

    run_metadata_path = DRIVE_OUTPUT_DIR / "embedding_run_metadata.json"
    atomic_write_json(
        run_metadata_path,
        {
            "model_name": model_name,
            "query_prompt": QUERY_PROMPT,
            "max_length": MAX_LENGTH,
            "embedding_dimension": dimension,
            "embedding_storage_dtype": EMBED_STORAGE_DTYPE,
            "effective_batch_size": encoder.batch_size,
            "item_fingerprint": item_fingerprint,
            "term_fingerprint": term_fingerprint,
            "train": train_summary,
            "submission": submission_summary,
            "elapsed_seconds": round(time.time() - started, 2),
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    upload_outputs([train_output, submission_output], run_metadata_path, model_name)
    print(f"HEPSI TAMAM ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
