"""ocr_engine_comparison.py — автономное сравнение OCR-движков на открытках.

Спайк для ветки ocr-extension: проверяет, насколько Tesseract, EasyOCR и
PaddleOCR читают текст на архивных открытках. В пайплайн не интегрировано —
цель только оценить пригодность движков на нескольких изображениях.

Движки опциональны: если библиотека не установлена, она помечается как
недоступная и пропускается (скрипт не падает). Распознавание идёт для
русского и английского одновременно.

Установка движков:
    # Tesseract (нужен системный бинарник + языковые пакеты)
    brew install tesseract tesseract-lang
    pip install pytesseract
    # EasyOCR
    pip install easyocr
    # PaddleOCR
    pip install paddlepaddle paddleocr

Запуск:
    PYTHONPATH=src python scripts/benchmarks/ocr_engine_comparison.py \\
        --images-dir data/ocr_eval
"""

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

# отключаем сетевую проверку источника моделей PaddleOCR (модели берутся из кэша)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


# языки распознавания в формате каждого движка
TESSERACT_LANGS = "rus+eng"
EASYOCR_LANGS = ["ru", "en"]
PADDLE_LANG = "ru"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="data/ocr_eval",
                   help="папка с изображениями для проверки")
    p.add_argument("--output", default="data/ocr_eval/ocr_comparison.json",
                   help="путь для JSON-результата")
    p.add_argument("--markdown", default="data/ocr_eval/ocr_comparison.md",
                   help="путь для читаемого Markdown-сравнения")
    p.add_argument("--engines", default="tesseract,easyocr,paddleocr",
                   help="список движков через запятую")
    p.add_argument("--limit", type=int, default=None,
                   help="ограничить число изображений (для больших коллекций)")
    p.add_argument("--shuffle", action="store_true",
                   help="случайно перемешать перед ограничением --limit")
    p.add_argument("--seed", type=int, default=42,
                   help="seed для воспроизводимого перемешивания")
    return p.parse_args()


def list_images(images_dir, limit=None, shuffle=False, seed=42):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in exts
    )
    if shuffle:
        import random
        random.Random(seed).shuffle(paths)
    if limit is not None:
        paths = paths[:limit]
    return paths


# ---------------------------------------------------------------------
# Движки: каждый класс лениво грузит модель и возвращает (текст, уверенность)
# ---------------------------------------------------------------------
class TesseractEngine:
    name = "tesseract"

    def __init__(self):
        import pytesseract  # noqa: F401
        self._pt = pytesseract
        self.version = str(pytesseract.get_tesseract_version())

    def run(self, image_path):
        from pytesseract import Output
        img = Image.open(image_path).convert("RGB")
        text = self._pt.image_to_string(img, lang=TESSERACT_LANGS).strip()
        if not text:
            # пустой результат — уверенность не имеет смысла
            return text, None
        # средняя уверенность по словам с conf > 0
        data = self._pt.image_to_data(img, lang=TESSERACT_LANGS, output_type=Output.DICT)
        confs = [float(c) for c in data["conf"] if c not in ("-1", -1)]
        confs = [c for c in confs if c >= 0]
        confidence = (sum(confs) / len(confs) / 100.0) if confs else None
        return text, confidence


class EasyOCREngine:
    name = "easyocr"

    def __init__(self):
        import easyocr
        # инициализация ридера долгая — делаем один раз
        self._reader = easyocr.Reader(EASYOCR_LANGS, gpu=False)
        self.version = "easyocr"

    def run(self, image_path):
        results = self._reader.readtext(str(image_path))
        # results: список (bbox, text, conf)
        lines = [r[1] for r in results]
        confs = [float(r[2]) for r in results]
        text = "\n".join(lines).strip()
        confidence = (sum(confs) / len(confs)) if confs else None
        return text, confidence


class PaddleOCREngine:
    name = "paddleocr"

    def __init__(self):
        from paddleocr import PaddleOCR
        # PaddleOCR 3.x: параметры use_angle_cls/show_log убраны из конструктора
        self._ocr = PaddleOCR(lang=PADDLE_LANG)
        self.version = "paddleocr"

    def run(self, image_path):
        # PaddleOCR 3.x: predict() возвращает список OCRResult с rec_texts/rec_scores
        result = self._ocr.predict(str(image_path))
        lines, confs = [], []
        for page in (result or []):
            texts = page.get("rec_texts", []) if hasattr(page, "get") else []
            scores = page.get("rec_scores", []) if hasattr(page, "get") else []
            lines.extend(texts)
            confs.extend(float(s) for s in scores)
        text = "\n".join(lines).strip()
        confidence = (sum(confs) / len(confs)) if confs else None
        return text, confidence


ENGINE_CLASSES = {
    "tesseract": TesseractEngine,
    "easyocr": EasyOCREngine,
    "paddleocr": PaddleOCREngine,
}


def init_engine(key):
    """Пытается инициализировать движок. Возвращает (engine, error)."""
    cls = ENGINE_CLASSES.get(key)
    if cls is None:
        return None, f"неизвестный движок {key!r}"
    try:
        return cls(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def render_markdown(images, results):
    L = ["# Сравнение OCR-движков на открытках\n"]
    for img in images:
        name = Path(img).name
        L.append(f"## {name}\n")
        for engine_key, engine_res in results.items():
            if not engine_res["available"]:
                L.append(f"### {engine_key} — недоступен ({engine_res['error']})\n")
                continue
            item = next((x for x in engine_res["per_image"] if x["image"] == img), None)
            if item is None or item.get("error"):
                err = item["error"] if item else "нет данных"
                L.append(f"### {engine_key} — ошибка: {err}\n")
                continue
            conf = item["confidence"]
            conf_s = f"{conf:.2f}" if conf is not None else "—"
            L.append(f"### {engine_key} (уверенность {conf_s}, {item['elapsed_sec']:.1f} с)\n")
            L.append("```\n" + (item["text"] or "(пусто)") + "\n```\n")
    return "\n".join(L)


def main():
    args = parse_args()
    images = list_images(args.images_dir, limit=args.limit,
                         shuffle=args.shuffle, seed=args.seed)
    if not images:
        print(f"[ERROR] нет изображений в {args.images_dir}")
        return
    print(f"[INFO] изображений: {len(images)}")

    engine_keys = [k.strip() for k in args.engines.split(",") if k.strip()]
    results = {}

    for key in engine_keys:
        print(f"\n[INFO] инициализация движка: {key}")
        engine, error = init_engine(key)
        if engine is None:
            print(f"  недоступен: {error}")
            results[key] = {"available": False, "error": error, "per_item": [], "per_image": []}
            continue

        per_image = []
        for img in images:
            img_str = str(img)
            try:
                t0 = time.time()
                text, confidence = engine.run(img_str)
                elapsed = time.time() - t0
                preview = text.replace("\n", " ⏎ ")[:80]
                print(f"  {Path(img).name}: {len(text)} симв., conf="
                      f"{confidence if confidence is None else round(confidence,2)}, "
                      f"{elapsed:.1f}с | {preview}")
                per_image.append({
                    "image": img_str,
                    "text": text,
                    "confidence": confidence,
                    "elapsed_sec": elapsed,
                    "error": None,
                })
            except Exception as e:
                print(f"  {Path(img).name}: ОШИБКА {type(e).__name__}: {e}")
                per_image.append({
                    "image": img_str, "text": "", "confidence": None,
                    "elapsed_sec": None, "error": f"{type(e).__name__}: {e}",
                })
        results[key] = {
            "available": True,
            "version": getattr(engine, "version", key),
            "error": None,
            "per_image": per_image,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"images": [str(i) for i in images], "engines": results},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[INFO] JSON сохранён: {out}")

    md = render_markdown([str(i) for i in images], results)
    md_path = Path(args.markdown)
    md_path.write_text(md, encoding="utf-8")
    print(f"[INFO] Markdown сохранён: {md_path}")


if __name__ == "__main__":
    main()
