# scripts/evaluate.py

import json
import re
import time
from pathlib import Path
from collections import Counter

import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.stem.snowball import SnowballStemmer

from sacrebleu.metrics import BLEU
from bert_score import score as bertscore_score

import numpy as np
import torch
from PIL import Image
import open_clip

import sys
sys.path.insert(0, "src")

from tsu_image_description.pipeline import ArchiveDescriptionPipeline


nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

ru_stemmer = SnowballStemmer("russian")


def load_references(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
    return text.strip()


def normalize_for_lexical_metrics(text: str) -> str:
    text = normalize_text(text)
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    stems = [ru_stemmer.stem(tok) for tok in tokens]
    return " ".join(stems)


def tokenize_for_lexical_metrics(text: str):
    return normalize_for_lexical_metrics(text).split()


def normalize_type_label(text: str) -> str:
    text = normalize_text(text)
    mapping = {
        "a postcard": "открытка",
        "postcard": "открытка",
        "открытка": "открытка",
        "greeting card": "открытка",
        "a greeting card": "открытка",
        "poster": "плакат",
        "a poster": "плакат",
        "плакат": "плакат",
        "illustration": "иллюстрация",
        "an illustration": "иллюстрация",
        "иллюстрация": "иллюстрация",
        "photograph": "фотография",
        "a photograph": "фотография",
        "photo": "фотография",
        "фотография": "фотография",
    }
    return mapping.get(text, text)


def encode_clip_image(image_path: str, model, preprocess, device: str) -> np.ndarray:
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.squeeze(0).detach().cpu().numpy().astype("float32")


def encode_clip_text(text: str, model, tokenizer, device: str) -> np.ndarray:
    text_tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.squeeze(0).detach().cpu().numpy().astype("float32")


def compute_clipscore_from_embeddings(image_emb: np.ndarray, text_emb: np.ndarray) -> float:
    return float(np.dot(image_emb, text_emb))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def rouge1_f1_from_tokens(pred_tokens, ref_tokens):
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())

    if len(pred_tokens) == 0 or len(ref_tokens) == 0 or overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def lcs_length(xs, ys):
    n = len(xs)
    m = len(ys)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        xi = xs[i - 1]
        for j in range(1, m + 1):
            if xi == ys[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m]


def rouge_l_f1_from_tokens(pred_tokens, ref_tokens):
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_text_metrics(predictions_raw_ru, references_raw_ru):
    """
    Лексические метрики считаем по нормализованным и стеммированным русским токенам.
    BERTScore считаем по сырым русским строкам.
    """
    bleu = BLEU(effective_order=True)

    pred_lex = [normalize_for_lexical_metrics(p) for p in predictions_raw_ru]
    ref_lex = [normalize_for_lexical_metrics(r) for r in references_raw_ru]

    pred_tok = [p.split() for p in pred_lex]
    ref_tok = [r.split() for r in ref_lex]

    meteor_scores = []
    rouge1_scores = []
    rougel_scores = []

    for pred_tokens, ref_tokens in zip(pred_tok, ref_tok):
        meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
        rouge1_scores.append(rouge1_f1_from_tokens(pred_tokens, ref_tokens))
        rougel_scores.append(rouge_l_f1_from_tokens(pred_tokens, ref_tokens))

    bleu_result = bleu.corpus_score(pred_lex, [[r for r in ref_lex]])

    # BERTScore по сырым русским строкам
    _, _, bert_f1 = bertscore_score(
        predictions_raw_ru,
        references_raw_ru,
        lang="ru",
        verbose=False,
    )

    return {
        "BLEU": float(bleu_result.score),
        "METEOR_mean": float(mean(meteor_scores)),
        "ROUGE-1-F1_mean": float(mean(rouge1_scores)),
        "ROUGE-L-F1_mean": float(mean(rougel_scores)),
        "BERTScore_F1_mean": float(bert_f1.mean().item()),
    }


def main():
    references_path = "data/eval/references.jsonl"
    refs = load_references(references_path)

    pipeline = ArchiveDescriptionPipeline()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    clip_model = clip_model.to(device).eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    short_predictions_raw_ru = []
    short_references_raw_ru = []

    clip_scores_pred_en = []
    clip_scores_pred_ru = []
    clip_scores_archive = []
    clip_scores_reference_ru = []

    total_time = 0.0
    rows = []

    image_embeddings = []
    reference_short_ru_embeddings = []
    predicted_caption_ru_embeddings = []
    predicted_caption_en_embeddings = []
    archive_prediction_embeddings = []

    for item in refs:
        image_path = item["image_path"]
        reference_short_ru = item["reference_short_ru"]
        reference_type = item.get("type", "")

        t0 = time.time()
        result = pipeline.run(image_path)
        elapsed = time.time() - t0

        pred_caption_ru_raw = result["caption"]["ru"]
        pred_caption_en_raw = result["caption"].get("en", "")
        pred_archive_raw = result["archive_description"]

        short_predictions_raw_ru.append(pred_caption_ru_raw)
        short_references_raw_ru.append(reference_short_ru)

        pred_type_raw = result["metadata"]["image_type"]["label"]
        pred_type_norm = normalize_type_label(pred_type_raw)
        ref_type_norm = normalize_type_label(reference_type) if reference_type else ""

        image_emb = encode_clip_image(
            image_path=image_path,
            model=clip_model,
            preprocess=clip_preprocess,
            device=device,
        )

        reference_short_ru_emb = encode_clip_text(
            text=reference_short_ru,
            model=clip_model,
            tokenizer=clip_tokenizer,
            device=device,
        )

        predicted_caption_ru_emb = encode_clip_text(
            text=pred_caption_ru_raw,
            model=clip_model,
            tokenizer=clip_tokenizer,
            device=device,
        )

        predicted_caption_en_text = pred_caption_en_raw if pred_caption_en_raw else pred_caption_ru_raw
        predicted_caption_en_emb = encode_clip_text(
            text=predicted_caption_en_text,
            model=clip_model,
            tokenizer=clip_tokenizer,
            device=device,
        )

        archive_prediction_emb = encode_clip_text(
            text=pred_archive_raw,
            model=clip_model,
            tokenizer=clip_tokenizer,
            device=device,
        )

        # Основной CLIPScore — по predicted English caption
        clip_val_pred_en = compute_clipscore_from_embeddings(image_emb, predicted_caption_en_emb)
        clip_val_pred_ru = compute_clipscore_from_embeddings(image_emb, predicted_caption_ru_emb)
        clip_val_archive = compute_clipscore_from_embeddings(image_emb, archive_prediction_emb)
        clip_val_reference_ru = compute_clipscore_from_embeddings(image_emb, reference_short_ru_emb)

        clip_scores_pred_en.append(clip_val_pred_en)
        clip_scores_pred_ru.append(clip_val_pred_ru)
        clip_scores_archive.append(clip_val_archive)
        clip_scores_reference_ru.append(clip_val_reference_ru)

        total_time += elapsed

        image_embeddings.append(image_emb)
        reference_short_ru_embeddings.append(reference_short_ru_emb)
        predicted_caption_ru_embeddings.append(predicted_caption_ru_emb)
        predicted_caption_en_embeddings.append(predicted_caption_en_emb)
        archive_prediction_embeddings.append(archive_prediction_emb)

        rows.append(
            {
                "embedding_row_idx": len(rows),
                "image_path": image_path,
                "caption_en_prediction": pred_caption_en_raw,
                "caption_ru_prediction": pred_caption_ru_raw,
                "reference_short_ru": reference_short_ru,
                "predicted_type_raw": pred_type_raw,
                "predicted_type_normalized": pred_type_norm,
                "reference_type": reference_type,
                "reference_type_normalized": ref_type_norm,
                "archive_prediction": pred_archive_raw,
                "clipscore_pred_en": clip_val_pred_en,
                "clipscore_pred_ru": clip_val_pred_ru,
                "clipscore_archive": clip_val_archive,
                "clipscore_reference_ru": clip_val_reference_ru,
                "clipscore": clip_val_pred_en,
                "latency_sec": elapsed,
            }
        )

    short_metrics = compute_text_metrics(
        short_predictions_raw_ru,
        short_references_raw_ru,
    )

    summary = {
        "num_examples": len(refs),
        "short_text_metrics": short_metrics,
        "CLIPScore_mean": float(mean(clip_scores_pred_en)),
        "CLIPScore_mean_pred_en": float(mean(clip_scores_pred_en)),
        "CLIPScore_mean_pred_ru": float(mean(clip_scores_pred_ru)),
        "CLIPScore_mean_archive": float(mean(clip_scores_archive)),
        "CLIPScore_mean_reference_ru": float(mean(clip_scores_reference_ru)),
        "Latency_mean_sec": float(total_time / len(refs)) if refs else 0.0,
        "Images_per_sec": float(len(refs) / total_time) if total_time > 0 else 0.0,
        "metric_protocol": {
            "BLEU_METEOR_ROUGE": "normalized russian text + Snowball stemming + custom token-based ROUGE",
            "BERTScore": "raw russian text",
            "main_CLIPScore_variant": "image ↔ predicted_caption_en",
            "type_metrics": "excluded from final summary as methodologically non-interpretable for current dataset",
        },
    }

    out_dir = Path("data/eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "predictions_detailed.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    np.savez_compressed(
        out_dir / "embeddings_clip.npz",
        image_embeddings=np.stack(image_embeddings),
        reference_short_ru_embeddings=np.stack(reference_short_ru_embeddings),
        predicted_caption_ru_embeddings=np.stack(predicted_caption_ru_embeddings),
        predicted_caption_en_embeddings=np.stack(predicted_caption_en_embeddings),
        archive_prediction_embeddings=np.stack(archive_prediction_embeddings),
    )

    with open(out_dir / "embeddings_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": "open_clip ViT-B-32",
                "pretrained": "openai",
                "normalized": True,
                "main_clipscore_variant": "image ↔ predicted_caption_en",
                "alignment_rule": (
                    "row i in predictions_detailed.json corresponds to row i "
                    "in every array in embeddings_clip.npz"
                ),
                "arrays": [
                    "image_embeddings",
                    "reference_short_ru_embeddings",
                    "predicted_caption_ru_embeddings",
                    "predicted_caption_en_embeddings",
                    "archive_prediction_embeddings",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== SHORT DESCRIPTION METRICS ===")
    for k, v in short_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== MULTIMODAL / CLIPSCORE ===")
    print(f"CLIPScore_mean (pred_en): {summary['CLIPScore_mean_pred_en']:.4f}")
    print(f"CLIPScore_mean (pred_ru): {summary['CLIPScore_mean_pred_ru']:.4f}")
    print(f"CLIPScore_mean (archive): {summary['CLIPScore_mean_archive']:.4f}")
    print(f"CLIPScore_mean (reference_ru): {summary['CLIPScore_mean_reference_ru']:.4f}")

    print("\n=== PERFORMANCE ===")
    print(f"Latency_mean_sec: {summary['Latency_mean_sec']:.4f}")
    print(f"Images_per_sec: {summary['Images_per_sec']:.4f}")

    print("\nSaved to:")
    print(out_dir / "metrics_summary.json")
    print(out_dir / "predictions_detailed.json")
    print(out_dir / "embeddings_clip.npz")
    print(out_dir / "embeddings_manifest.json")


if __name__ == "__main__":
    main()
