"""upgrade_neb_resolution.py — апгрейд миниатюр НЭБ до полного разрешения.

Миниатюры (thumbnail.php, ~315×460) заменяются полноразмерным сканом из PDF
(getFiles.php?book_id=...&doc_type=pdf). PDF транзитный: скачали -> извлекли
лицевую сторону (страницу, совпадающую с миниатюрой по перцептивной подписи)
-> сохранили JPEG поверх файла -> PDF в памяти, на диск не пишется.

Идемпотентно: файлы, у которых ширина уже >= --skip-existing-width, пропускаются,
поэтому повторный запуск продолжает с места обрыва. Источники — любые jsonl/json
с полями image_path и catalog_id:
  - data/eval/references_neb_n224.jsonl        (НЭБ, ВОВ)
  - data/neb_search_postcards/manifest.json    (НЭБ, общий)

Использование:
    python scripts/data_preparation/upgrade_neb_resolution.py \
        --inputs data/eval/references_neb_n224.jsonl data/neb_search_postcards/manifest.json \
        --quality 90 --delay 1.0
"""

import argparse
import json
import time
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import requests
from PIL import Image

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
GETFILES = "https://rusneb.ru/local/tools/exalead/getFiles.php?book_id={cid}&doc_type=pdf"


def load_pairs(path):
    """(image_path, catalog_id) из jsonl или json-манифеста (dict или list)."""
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows = data
        else:  # dict: записи в items/records либо среди значений-словарей
            rows = data.get("items") or data.get("records") \
                or [v for v in data.values() if isinstance(v, dict)]
    return [(r["image_path"], r["catalog_id"]) for r in rows
            if isinstance(r, dict) and r.get("image_path") and r.get("catalog_id")]


def _sig(im):
    """Перцептивная подпись 64×64 grayscale — для выбора нужной страницы PDF."""
    return np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)


def best_page_image(pdf_bytes, thumb):
    """Страница PDF, наиболее похожая на миниатюру (лицевая сторона). -> (mse, Image)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    ts = _sig(thumb)
    best = None
    for pno in range(len(doc)):
        for x in doc[pno].get_images(full=True):
            im = Image.open(BytesIO(doc.extract_image(x[0])["image"])).convert("RGB")
            mse = float(((_sig(im) - ts) ** 2).mean())
            if best is None or mse < best[0]:
                best = (mse, im)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--min-upscale", type=float, default=1.5,
                    help="Сохранять, только если ширина выросла хотя бы в N раз.")
    ap.add_argument("--skip-existing-width", type=int, default=700,
                    help="Пропускать файлы, у которых ширина уже >= этого (уже апгрейжены).")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    seen, pairs = set(), []
    for inp in args.inputs:
        for ip, cid in load_pairs(inp):
            if ip not in seen:
                seen.add(ip)
                pairs.append((ip, cid))
    print(f"[upgrade] всего к обработке: {len(pairs)}")

    sess = requests.Session()
    sess.headers.update(UA)
    up = skip = fail = 0

    for i, (ip, cid) in enumerate(pairs, 1):
        path = Path(ip)
        if not path.is_file():
            print(f"[{i}/{len(pairs)}] НЕТ ФАЙЛА {ip}")
            fail += 1
            continue
        try:
            cur = Image.open(path)
            cw = cur.size[0]
        except Exception:
            cur, cw = None, 0
        if cw >= args.skip_existing_width:
            skip += 1
            continue

        try:
            resp = None
            for a in range(args.retries):
                try:
                    resp = sess.get(GETFILES.format(cid=cid), timeout=90)
                    resp.raise_for_status()
                    break
                except Exception:
                    if a == args.retries - 1:
                        raise
                    time.sleep(2 * (a + 1))
            res = best_page_image(resp.content, cur or Image.new("RGB", (1, 1)))
            if not res:
                print(f"[{i}/{len(pairs)}] нет изображений в PDF {cid}")
                fail += 1
                continue
            mse, im = res
            if cur and im.size[0] < cw * args.min_upscale:
                print(f"[{i}/{len(pairs)}] {path.name}: прирост мал ({cur.size}->{im.size}), пропуск")
                skip += 1
                continue
            im.save(path, "JPEG", quality=args.quality, optimize=True)
            up += 1
            print(f"[{i}/{len(pairs)}] {path.name}: "
                  f"{cur.size if cur else '?'} -> {im.size} "
                  f"{path.stat().st_size // 1024}КБ (mse={mse:.0f})")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(pairs)}] ОШИБКА {cid}: {e}")
        time.sleep(args.delay)

    print(f"\n[upgrade] апгрейжено={up} пропущено={skip} ошибок={fail}")


if __name__ == "__main__":
    main()
