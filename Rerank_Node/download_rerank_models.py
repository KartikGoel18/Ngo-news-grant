"""Download reranker models into the project-local cache.

Run this once before production/runtime execution so `rerank_node.py` can load
models from disk instead of contacting Hugging Face.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RERANK_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CACHE_DIR = RERANK_DIR / "model_cache"
DEFAULT_BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BI_ENCODER_CACHE_NAME = "bi_encoder"
CROSS_ENCODER_CACHE_NAME = "cross_encoder"


def _save_sentence_transformer(model, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save"):
        model.save(str(output_dir))
        return
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output_dir))
        return
    raise TypeError(f"Model object {type(model).__name__} does not expose save/save_pretrained.")


def download_rerank_models(
        cache_dir: Path,
        bi_encoder_model: str,
        cross_encoder_model: str,
        device: str,
        verify: bool,
) -> dict:
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "sentence-transformers is required to download reranker models. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from e

    cache_dir.mkdir(parents=True, exist_ok=True)
    bi_encoder_dir = cache_dir / BI_ENCODER_CACHE_NAME
    cross_encoder_dir = cache_dir / CROSS_ENCODER_CACHE_NAME

    print(f"Downloading bi-encoder: {bi_encoder_model}")
    bi_encoder = SentenceTransformer(bi_encoder_model, device=device)
    _save_sentence_transformer(bi_encoder, bi_encoder_dir)
    print(f"Saved bi-encoder to: {bi_encoder_dir}")

    print(f"Downloading cross-encoder: {cross_encoder_model}")
    cross_encoder = CrossEncoder(cross_encoder_model, device=device)
    _save_sentence_transformer(cross_encoder, cross_encoder_dir)
    print(f"Saved cross-encoder to: {cross_encoder_dir}")

    manifest = {
        "bi_encoder_model": bi_encoder_model,
        "bi_encoder_path": str(bi_encoder_dir),
        "cross_encoder_model": cross_encoder_model,
        "cross_encoder_path": str(cross_encoder_dir),
    }
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if verify:
        print("Verifying local model loads...")
        SentenceTransformer(str(bi_encoder_dir), device=device)
        CrossEncoder(str(cross_encoder_dir), device=device)
        print("Verification complete.")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predownload reranker models into Rerank_Node/model_cache.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_MODEL_CACHE_DIR))
    parser.add_argument("--bi-encoder-model", default=DEFAULT_BI_ENCODER_MODEL)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = download_rerank_models(
        cache_dir=Path(args.cache_dir),
        bi_encoder_model=args.bi_encoder_model,
        cross_encoder_model=args.cross_encoder_model,
        device=args.device,
        verify=not args.no_verify,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
