"""eval_reference_based.py — reference-based metrics для русских описаний.

Применяется к выходу `scripts/evaluate.py` (metrics_*.json с per_item).
Считает supervised-метрики против curator-references:
  - BERTScore_RU (semantic similarity)
  - sacreBLEU (n-gram corpus-level)
  - chrF (character-level, толерантна к морфологии)
  - ROUGE-L (longest common subsequence F1)
  - Длина output vs reference (median ratio)

Поддерживает сравнение нескольких полей (archive_ru как primary; caption_ru как secondary).

Использование:
    python scripts/eval_reference_based.py \\
      --input data/eval/results/metrics_E12_neb_n224.json \\
      --output data/eval/results/refbased_E12_neb_n224.json \\
      --fields archive_ru,caption_ru
"""

import argparse
import json
import statistics
from pathlib import Path

import sacrebleu
from rouge_score import rouge_scorer


def load_pairs(input_path, field, ref_key="reference_ru"):
    """Читает metrics JSON, возвращает список (ref, hyp, image_path)."""
    with open(input_path) as f:
        d = json.load(f)
    per_item = d.get("per_item", [])
    pairs = []
    for it in per_item:
        ref = (it.get(ref_key) or it.get("reference_short_ru") or "").strip()
        hyp = (it.get(field) or "").strip()
        if ref and hyp:
            pairs.append((ref, hyp, it.get("image_path", "")))
    return pairs, d


def compute_bertscore(refs, hyps, lang="ru", model_type=None, batch_size=16):
    """Возвращает dict со средними P/R/F1 и per-item F1."""
    from bert_score import score as bert_score
    kw = dict(lang=lang, batch_size=batch_size, verbose=False, rescale_with_baseline=False)
    if model_type:
        kw["model_type"] = model_type
    P, R, F1 = bert_score(hyps, refs, **kw)
    return {
        "mean_precision": float(P.mean()),
        "mean_recall": float(R.mean()),
        "mean_f1": float(F1.mean()),
        "per_item_f1": F1.tolist(),
    }


def compute_sacrebleu(refs, hyps):
    """BLEU и chrF на уровне корпуса (стандартные конфиги sacrebleu)."""
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "bleu_details": str(bleu),
        "chrf_details": str(chrf),
    }


class _CyrillicWordTokenizer:
    """Токенизатор для rouge-score с поддержкой кириллицы. Штатный токенизатор
    rouge-score выбрасывает не-ASCII символы через NON_ALPHANUM_RE — для русского
    текста это даёт 0. Здесь разбиваем по границам слов, сохраняя буквы Unicode,
    и приводим к нижнему регистру для регистронезависимого сравнения."""

    _RE = None

    def __init__(self):
        import re
        # \w с re.UNICODE захватывает кириллицу
        self._RE = re.compile(r"[\w]+", re.UNICODE)

    def tokenize(self, text):
        return [t for t in self._RE.findall(text.lower()) if t]


def compute_rouge(refs, hyps):
    """ROUGE-L F1 по каждой паре, затем среднее. Токенизатор с поддержкой Unicode."""
    tokenizer = _CyrillicWordTokenizer()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=tokenizer)
    f1s = []
    for ref, hyp in zip(refs, hyps):
        s = scorer.score(ref, hyp)
        f1s.append(s["rougeL"].fmeasure)
    return {
        "mean_rouge_l_f1": float(statistics.mean(f1s)),
        "median_rouge_l_f1": float(statistics.median(f1s)),
        "per_item_f1": f1s,
    }


def compute_length_stats(refs, hyps):
    ref_lens = [len(r.split()) for r in refs]
    hyp_lens = [len(h.split()) for h in hyps]
    ratios = [h / r for h, r in zip(hyp_lens, ref_lens) if r > 0]
    return {
        "n": len(refs),
        "ref_len_median": statistics.median(ref_lens),
        "ref_len_mean": statistics.mean(ref_lens),
        "hyp_len_median": statistics.median(hyp_lens),
        "hyp_len_mean": statistics.mean(hyp_lens),
        "length_ratio_median": statistics.median(ratios) if ratios else None,
    }


def evaluate_field(input_path, field, lang="ru", bertscore_model=None):
    print(f"\n=== Field: {field} ===")
    pairs, full = load_pairs(input_path, field)
    if not pairs:
        print(f"  no pairs available (field '{field}' empty)")
        return None
    refs = [p[0] for p in pairs]
    hyps = [p[1] for p in pairs]
    images = [p[2] for p in pairs]
    print(f"  n pairs: {len(pairs)}")

    print("  computing length stats...")
    length = compute_length_stats(refs, hyps)

    print("  computing sacreBLEU + chrF...")
    bleu_chrf = compute_sacrebleu(refs, hyps)

    print("  computing ROUGE-L...")
    rouge = compute_rouge(refs, hyps)

    print("  computing BERTScore_RU...")
    bertscore = compute_bertscore(refs, hyps, lang=lang, model_type=bertscore_model)

    per_item = []
    for ref, hyp, img, b_f1, r_f1 in zip(
        refs, hyps, images, bertscore["per_item_f1"], rouge["per_item_f1"]
    ):
        per_item.append({
            "image_path": img,
            "reference": ref,
            "hypothesis": hyp,
            "bertscore_f1": b_f1,
            "rouge_l_f1": r_f1,
        })

    result = {
        "field": field,
        "length_stats": length,
        "bleu_chrf": bleu_chrf,
        "rouge_l": {k: v for k, v in rouge.items() if k != "per_item_f1"},
        "bertscore_ru": {k: v for k, v in bertscore.items() if k != "per_item_f1"},
        "per_item": per_item,
    }
    print(f"  BERTScore_RU F1: {bertscore['mean_f1']:.4f}")
    print(f"  ROUGE-L F1     : {rouge['mean_rouge_l_f1']:.4f}")
    print(f"  BLEU           : {bleu_chrf['bleu']:.2f}")
    print(f"  chrF           : {bleu_chrf['chrf']:.2f}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="metrics_*.json from evaluate.py")
    p.add_argument("--output", required=True)
    p.add_argument("--fields", default="archive_ru",
                   help="Comma-separated fields to evaluate (e.g. archive_ru,caption_ru)")
    p.add_argument("--lang", default="ru")
    p.add_argument("--bertscore-model", default=None,
                   help="Override BERTScore model (default: from lang)")
    return p.parse_args()


def main():
    args = parse_args()
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Fields: {fields}")

    results = {}
    for field in fields:
        r = evaluate_field(args.input, field, lang=args.lang,
                           bertscore_model=args.bertscore_model)
        if r:
            results[field] = r

    # сводка для сравнения полей между собой
    summary = {}
    for field, r in results.items():
        summary[field] = {
            "n": r["length_stats"]["n"],
            "bertscore_f1": r["bertscore_ru"]["mean_f1"],
            "rouge_l_f1": r["rouge_l"]["mean_rouge_l_f1"],
            "bleu": r["bleu_chrf"]["bleu"],
            "chrf": r["bleu_chrf"]["chrf"],
            "length_ratio_median": r["length_stats"]["length_ratio_median"],
        }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Field':<15}  {'n':>4}  {'BERTScore':>10}  {'ROUGE-L':>10}  {'BLEU':>7}  {'chrF':>7}  {'len_ratio':>10}")
    for f, s in summary.items():
        print(f"{f:<15}  {s['n']:>4}  {s['bertscore_f1']:>10.4f}  "
              f"{s['rouge_l_f1']:>10.4f}  {s['bleu']:>7.2f}  {s['chrf']:>7.2f}  "
              f"{s['length_ratio_median']:>10.2f}")
    print()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "input": args.input,
            "fields": fields,
            "summary": summary,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
