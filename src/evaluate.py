import json
from typing import List, Dict

import nltk
import sacrebleu
from bert_score import score as bertscore_score
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from tsu_image_description.models import CaptionGenerator, Translator


nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def compute_lexical_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rougeL_scores = []
    meteor_scores = []

    for pred, ref in zip(predictions, references):
        rouge_result = rouge.score(ref, pred)
        rouge1_scores.append(rouge_result["rouge1"].fmeasure)
        rougeL_scores.append(rouge_result["rougeL"].fmeasure)

        meteor_scores.append(
            meteor_score(
                [nltk.word_tokenize(ref.lower())],
                nltk.word_tokenize(pred.lower())
            )
        )

    bleu = sacrebleu.corpus_bleu(predictions, [references])

    return {
        "BLEU": round(bleu.score, 4),
        "ROUGE-1": round(sum(rouge1_scores) / len(rouge1_scores), 4),
        "ROUGE-L": round(sum(rougeL_scores) / len(rougeL_scores), 4),
        "METEOR": round(sum(meteor_scores) / len(meteor_scores), 4),
    }


def compute_semantic_metrics(predictions_ru: List[str], references_ru: List[str]) -> Dict[str, float]:
    _, _, f1 = bertscore_score(
        predictions_ru,
        references_ru,
        lang="ru",
        verbose=False
    )

    return {
        "BERTScore-F1": round(float(f1.mean().item()), 4),
    }


def evaluate(dataset_path: str) -> Dict:
    data = load_jsonl(dataset_path)

    caption_model = CaptionGenerator()
    translator = Translator()

    predictions_ru = []
    references_ru = []
    sample_results = []

    for item in data:
        image_path = item["image_path"]
        reference_ru = item["reference_ru"]

        pred_en = caption_model.generate(image_path)
        pred_ru = translator.translate(pred_en)

        predictions_ru.append(pred_ru)
        references_ru.append(reference_ru)

        sample_results.append({
            "image_path": image_path,
            "reference_ru": reference_ru,
            "prediction_ru": pred_ru,
            "prediction_en": pred_en,
        })

        print(f"\nImage: {image_path}")
        print(f"REF_RU:  {reference_ru}")
        print(f"PRED_RU: {pred_ru}")
        print(f"PRED_EN: {pred_en}")

    lexical_metrics = compute_lexical_metrics(predictions_ru, references_ru)
    semantic_metrics = compute_semantic_metrics(predictions_ru, references_ru)

    return {
        "num_samples": len(data),
        "lexical_metrics": lexical_metrics,
        "semantic_metrics": semantic_metrics,
        "samples": sample_results,
    }


if __name__ == "__main__":
    dataset_path = "data/eval/references.jsonl"
    results = evaluate(dataset_path)

    print("\n=== FINAL RESULTS ===")
    print(json.dumps({
        "num_samples": results["num_samples"],
        "lexical_metrics": results["lexical_metrics"],
        "semantic_metrics": results["semantic_metrics"],
    }, ensure_ascii=False, indent=2))
