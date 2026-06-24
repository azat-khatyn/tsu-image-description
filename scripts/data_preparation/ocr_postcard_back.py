"""ocr_postcard_back.py — распознавание ОБОРОТА открытки НЭБ и разбор на поля.

Оборот открытки несёт фактологию каталога (заголовок, редактор, издательство,
номер, год) — то, чего нет в визуальном описании лицевой стороны. Скрипт:
  1. качает PDF открытки (getFiles.php), берёт страницу-оборот
     (ту, что НЕ совпадает с лицевой миниатюрой);
  2. распознаёт текст (PaddleOCR);
  3. фильтрует почтовый бойлерплейт и разбирает на поля.

Зацепка парсинга: на советских открытках заголовок идёт прямо перед «Редактор»,
а издательство/номер — после.

Использование:
    python scripts/data_preparation/ocr_postcard_back.py \
        --references data/eval/references_neb_n224.jsonl --limit 20 \
        --out data/eval/neb_back_ocr.jsonl
"""

import argparse
import json
import re
import tempfile
from io import BytesIO

# Почтовый бойлерплейт оборота (с допуском на искажения OCR кириллицы/латиницы).
_BOILERPLATE = [
    r"почтова\w*\s*карточк\w*", r"м[еe]ст[оo]\s*[фf]?\s*для\s*марк\w*",
    # шапка-бойлерплейт «ОТЕЧЕСТВЕННАЯ ВОЙНА» (именительный) — но НЕ «Отечественной
    # войны» (родительный) внутри заголовка, его сохраняем.
    r"отечественная\s+война", r"открыт\w*\s*письм\w*",
    r"адрес\s*отправ\w*", r"\bкуда\b", r"\bк[оo]м[уy]\b",
]
_CODES = [
    r"\b\d{1,2}[-\s]?\d{3,4}\s*/\s*\d{1,4}\w*",   # 3-3100/7, 13-3100/99
    r"\bМ[-\s]?\d{3,6}\b", r"\bH[-\s]?\w{2,6}/\d\b",
    r"\b\d{2,3}\.\d{3,5}\b", r"\b\d{1,3}/\d{1,3}\b",
]
_PUBLISHERS = ["Искусство", "ИЗОГИЗ", "Советский художник", "Аврора"]


def parse_back(raw: str) -> dict:
    """Разбор сырого OCR оборота на поля каталога (эвристика)."""
    editor = None
    m = re.search(r"[РP]едактор[\s:.]*([А-ЯA-Z]\.?\s?[А-ЯA-Z]\.?\s?[А-ЯЁ][а-яёА-ЯЁ]{3,})", raw)
    if m:
        editor = re.sub(r"\s+", " ", m.group(1)).strip(" .,")

    number = None
    m = re.search(r"№\s*([0-9]{2,6})", raw)
    if m:
        number = m.group(1)

    publisher = next((p for p in _PUBLISHERS if p.lower() in raw.lower()), None)

    year = None
    m = re.search(r"\b(1[789]\d{2}(?:\s*[-–]\s*19\d{2})?)\b", raw)
    if m:
        year = re.sub(r"\s+", "", m.group(1))

    # Заголовок: между концом адресной части и «Редактор». Адресная часть
    # завершается на «Кому» (терпимо к Latin-OCR: Koмy и т.п.); заголовок — после.
    pre = re.split(r"[РP]едактор", raw)[0]
    anchors = list(re.finditer(r"[КK][оoO][мmМ][уyY]", pre))
    if anchors:
        pre = pre[anchors[-1].end():]
    else:  # запасной путь — снять явный бойлерплейт
        for pat in _BOILERPLATE:
            pre = re.sub(pat, " ", pre, flags=re.I)
    for pat in _CODES:
        pre = re.sub(pat, " ", pre)
    pre = re.sub(r"[=*•·]+", " ", pre)
    pre = re.sub(r"\s+", " ", pre)
    # отрезаем ведущий мусор до первой кириллической буквы (коды, цифры, латиница)
    title = re.sub(r"^[^А-ЯЁа-яё]+", "", pre).strip(" .,;:-") or None
    if title and len(title) < 4:
        title = None

    return {"title": title, "editor": editor, "publisher": publisher,
            "number": number, "year": year}


def _sig(im):
    import numpy as np
    return np.asarray(im.convert("L").resize((64, 64)), dtype="float32")


def back_image(cid, thumb, session):
    """Страница PDF-оборота (наиболее далёкая от лицевой миниатюры). -> PIL или None."""
    import fitz
    from PIL import Image
    url = f"https://rusneb.ru/local/tools/exalead/getFiles.php?book_id={cid}&doc_type=pdf"
    pdf = session.get(url, timeout=90).content
    doc = fitz.open(stream=pdf, filetype="pdf")
    pages = [Image.open(BytesIO(doc.extract_image(x[0])["image"])).convert("RGB")
             for p in range(len(doc)) for x in doc[p].get_images(full=True)]
    if len(pages) < 2:
        return None
    ts = _sig(thumb)
    order = sorted(range(len(pages)), key=lambda i: ((_sig(pages[i]) - ts) ** 2).mean())
    return pages[order[-1]]


def main():
    import requests
    from PIL import Image
    from tsu_image_description.ocr_extractor import OCRExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--references", default="data/eval/references_neb_n224.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default="data/eval/neb_back_ocr.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.references, encoding="utf-8")][: args.limit]
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                       "AppleWebKit/537.36 Chrome/120 Safari/537.36"})
    ocr = OCRExtractor()

    out_f = open(args.out, "w", encoding="utf-8")
    n_ok = n_skip = 0
    for r in rows:
        cid = r["catalog_id"]
        try:
            back = back_image(cid, Image.open(r["image_path"]), sess)
        except Exception as e:
            print(f"[{cid}] ошибка загрузки: {e}"); n_skip += 1; continue
        if back is None:
            print(f"[{cid}] оборота нет (одна страница)"); n_skip += 1; continue
        tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        back.save(tf.name, "JPEG", quality=92)
        res = ocr.extract(tf.name)
        raw = res.get("raw_text") or res.get("text") or ""
        fields = parse_back(raw)
        rec = {"catalog_id": cid, "back_fields": fields,
               "ocr_confidence": res.get("confidence"), "raw_ocr": raw}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        n_ok += 1
        print(f"[{cid}] заголовок: {fields['title']!r} | ред.: {fields['editor']} "
              f"| изд.: {fields['publisher']} | №: {fields['number']}")
    out_f.close()
    print(f"\nготово: {n_ok} оборотов -> {args.out}; пропущено {n_skip}")


if __name__ == "__main__":
    main()
