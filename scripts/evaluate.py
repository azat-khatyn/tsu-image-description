# scripts/evaluate.py

import json
import re
import time
from pathlib import Path

import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bertscore_score

import torch
from PIL import Image
import open_clip

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import sys
sys.path.insert(0, "src")

from tsu_image_description.pipeline import ArchiveDescriptionPipeline


nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


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


def compute_clipscore(image_path: str, text: str, model, preprocess, tokenizer, device: str) -> float:
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text_tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        sim = (image_features @ text_features.T).item()

    return float(sim)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def compute_text_metrics(predictions, references):
    bleu = BLEU(effective_order=True)
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    meteor_scores = []
    rouge1_scores = []
    rougel_scores = []

    for pred, ref in zip(predictions, references):
        meteor_scores.append(meteor_score([ref.split()], pred.split()))
        rouge_scores = rouge.score(ref, pred)
        rouge1_scores.append(rouge_scores["rouge1"].fmeasure)
        rougel_scores.append(rouge_scores["rougeL"].fmeasure)

    bleu_result = bleu.corpus_score(predictions, [[r for r in references]])
    _, _, bert_f1 = bertscore_score(predictions, references, lang="ru", verbose=False)

    return {
        "BLEU": float(bleu_result.score),
        "METEOR_mean": float(mean(meteor_scores)),
        "ROUGE-1-F1_mean": float(mean(rouge1_scores)),
        "ROUGE-L-F1_mean": float(mean(rougel_scores)),
        "BERTScore_F1_mean": float(bert_f1.mean().item()),
    }


def compute_type_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
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

    short_predictions = []
    short_references = []

    type_true = []
    type_pred = []

    clip_scores = []
    total_time = 0.0
    rows = []

    for item in refs:
        image_path = item["image_path"]
        reference_short_ru = item["reference_short_ru"]
        reference_type = item["type"]

        t0 = time.time()
        result = pipeline.run(image_path)
        elapsed = time.time() - t0

        pred_caption_ru_raw = result["caption"]["ru"]
        pred_archive_raw = result["archive_description"]

        pred_caption_ru = normalize_text(pred_caption_ru_raw)
        ref_short_ru = normalize_text(reference_short_ru)

        short_predictions.append(pred_caption_ru)
        short_references.append(ref_short_ru)

        pred_type_raw = result["metadata"]["image_type"]["label"]
        pred_type_norm = normalize_type_label(pred_type_raw)
        ref_type_norm = normalize_type_label(reference_type)

        type_true.append(ref_type_norm)
        type_pred.append(pred_type_norm)

        clip_val = compute_clipscore(
            image_path=image_path,
            text=pred_archive_raw,
            model=clip_model,
            preprocess=clip_preprocess,
            tokenizer=clip_tokenizer,
            device=device,
        )
        clip_scores.append(clip_val)
        total_time += elapsed

        rows.append(
            {
                "image_path": image_path,
                "caption_ru_prediction": pred_caption_ru_raw,
                "reference_short_ru": reference_short_ru,
                "predicted_type_raw": pred_type_raw,
                "predicted_type_normalized": pred_type_norm,
                "reference_type": reference_type,
                "reference_type_normalized": ref_type_norm,
                "archive_prediction": pred_archive_raw,
                "clipscore": clip_val,
                "latency_sec": elapsed,
            }
        )

    short_metrics = compute_text_metrics(short_predictions, short_references)
    type_metrics = compute_type_metrics(type_true, type_pred)

    summary = {
        "num_examples": len(refs),
        "short_text_metrics": short_metrics,
        "type_metrics": type_metrics,
        "CLIPScore_mean": float(mean(clip_scores)),
        "Latency_mean_sec": float(total_time / len(refs)) if refs else 0.0,
        "Images_per_sec": float(len(refs) / total_time) if total_time > 0 else 0.0,
    }

    out_dir = Path("data/eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "predictions_detailed.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n=== SHORT DESCRIPTION METRICS ===")
    for k, v in short_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== TYPE CLASSIFICATION METRICS ===")
    for k, v in type_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== ARCHIVE DESCRIPTION / MULTIMODAL ===")
    print(f"CLIPScore_mean: {summary['CLIPScore_mean']:.4f}")

    print("\n=== PERFORMANCE ===")
    print(f"Latency_mean_sec: {summary['Latency_mean_sec']:.4f}")
    print(f"Images_per_sec: {summary['Images_per_sec']:.4f}")

    print("\nSaved to:")
    print(out_dir / "metrics_summary.json")
    print(out_dir / "predictions_detailed.json")


if __name__ == "__main__":
    main()
