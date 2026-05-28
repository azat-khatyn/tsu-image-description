"""scrape_neb_collection.py — собрать коллекцию открыток с НЭБ.

Парсит каталожные карточки rusneb.ru, извлекает метаданные RUSMARC
и скачивает изображения-миниатюры для дальнейшего использования
в эталонной оценке.

Текущая целевая коллекция:
    529 «Ленинград в Великой Отечественной войне 1941-1945 гг. (открытки)»
    https://rusneb.ru/collections/529_leningrad_v_velikoy_otechestvennoy_voyne_1941_1945_gg_otkrytki/
    324 материала

Извлекаемые поля RUSMARC:
    001         — catalog_id
    200 $a/$e/$f — заглавие / формат / художник
    210 $a/$c/$d — место / издатель / год
    215 $a/$c/$d — объём / техника / размеры
    321 $a      — источник публикации (если упомянут в каталоге)
    327 $a      — примечание содержания
    540 $a      — вариант заглавия и именованные сущности (может быть несколько)
    606 $a/$x/$z — тематическая рубрика / подрубрика / дата
    700 $a/$b/$f — автор: фамилия / инициалы / годы жизни
    852 $b      — шифр хранения
    856 $u      — внешние URL (vivaldi.nlr.ru/...)

Изображения: миниатюра 315×460 JPEG (полное разрешение скрыто за JS-viewer).
URL: https://rusneb.ru/local/tools/exalead/thumbnail.php?url={catalog_id}

Использование:
    python scripts/scrape_neb_collection.py \\
        --output-dir data/neb_leningrad_wwii \\
        --delay 1.0
    # быстрая проверка:
    python scripts/scrape_neb_collection.py \\
        --output-dir data/neb_leningrad_wwii \\
        --max-items 5
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

COLLECTION_URL = (
    "https://rusneb.ru/collections/"
    "529_leningrad_v_velikoy_otechestvennoy_voyne_1941_1945_gg_otkrytki/"
)
BASE = "https://rusneb.ru"
THUMBNAIL_TEMPLATE = (
    BASE + "/local/tools/exalead/thumbnail.php?url={catalog_id}"
)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36 "
    "(research-project TSU-thesis archival-image-captioning)"
)


def get(session, url, timeout=30, retries=3):
    """GET с повторами и вежливым User-Agent."""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  ! retry {attempt+1}/{retries} after error: {e}")
            time.sleep(2 ** attempt)
    return None  # недостижимо


# ---------------------------------------------------------------------------
# Перебор страниц
# ---------------------------------------------------------------------------

def collect_catalog_ids(session, collection_url, delay=1.0, max_pages=50):
    """Обойти страницы коллекции и собрать ссылки /catalog/{id}/."""
    catalog_ids = []
    seen = set()

    for page_num in range(1, max_pages + 1):
        url = f"{collection_url}?page=page-{page_num}"
        print(f"[page {page_num}] {url}")
        try:
            r = get(session, url)
        except requests.RequestException as e:
            print(f"  ! cannot fetch page {page_num}: {e}; stopping")
            break

        page_ids = re.findall(r'/catalog/([A-Za-z0-9_]+)/', r.text)
        n_new = 0
        for cid in page_ids:
            if cid in seen:
                continue
            seen.add(cid)
            catalog_ids.append(cid)
            n_new += 1
        if n_new == 0:
            print("  ← no new catalog IDs, stop pagination")
            break
        print(f"  found {n_new} new (total {len(catalog_ids)})")
        time.sleep(delay)

    return catalog_ids


# ---------------------------------------------------------------------------
# Разбор RUSMARC из HTML карточки
# ---------------------------------------------------------------------------

SUBFIELD_RE = re.compile(r"\$([a-z0-9]):\s*(.*?)(?=(?:<br\s*/?>\s*)?\$[a-z0-9]:|$)",
                         flags=re.IGNORECASE | re.DOTALL)


def parse_rusmarc_subfields(html_chunk):
    """Разобрать фрагмент '$a: foo<br>$b: bar' в словарь {a: 'foo', b: 'bar'}.

    Повторяющиеся подполя объединяются через ' | '.
    Убираются теги <br> и хвостовые пробелы.
    """
    # убрать ведущий индикатор (например "1#" или "##" перед первым $)
    cleaned = re.sub(r"<br\s*/?>", "\n", html_chunk)
    out = {}
    for m in SUBFIELD_RE.finditer(cleaned):
        code = m.group(1).lower()
        value = m.group(2).strip().rstrip("<br>").rstrip().rstrip(";").strip()
        # удалить остатки HTML
        value = re.sub(r"\s+", " ", value).strip()
        if code in out:
            out[code] = out[code] + " | " + value
        else:
            out[code] = value
    return out


def parse_catalog_page(html: str, catalog_id: str) -> dict:
    """Извлечь структурированную запись из HTML карточки НЭБ."""
    soup = BeautifulSoup(html, "html.parser")

    record = {
        "catalog_id": catalog_id,
        "source_url": f"{BASE}/catalog/{catalog_id}/",
        "title": None,
        "variant_titles": [],
        "content_notes": [],
        "format": None,
        "artist": None,
        "place": None,
        "publisher": None,
        "year": None,
        "technique": None,
        "quantity": None,
        "dimensions": None,
        "shelf_mark": None,
        "topical_subjects": [],
        "topical_subjects_dates": [],
        "authors": [],
        "external_urls": [],
        "vivaldi_id": None,
    }

    # разбор таблицы RUSMARC — эти строки надёжны
    rows = soup.select("div.cards-table__row")
    by_tag = {}
    for row in rows:
        left = row.select_one(".cards-table__left")
        right = row.select_one(".cards-table__right")
        if not (left and right):
            continue
        tag = left.get_text(strip=True)
        chunk = right.decode_contents()
        # сохранить <br> для разбиения на подполя
        by_tag.setdefault(tag, []).append(chunk)

    def fields(tag):
        return by_tag.get(tag, [])

    # 200 — блок заглавия
    for chunk in fields("200"):
        sub = parse_rusmarc_subfields(chunk)
        if "a" in sub and not record["title"]:
            record["title"] = sub["a"]
        if "e" in sub:
            record["format"] = sub["e"]
        if "f" in sub and not record["artist"]:
            # "худ.: Гордон М.А."
            record["artist"] = sub["f"]

    # 210 — блок издания
    for chunk in fields("210"):
        sub = parse_rusmarc_subfields(chunk)
        if "a" in sub:
            record["place"] = sub["a"]
        if "c" in sub:
            record["publisher"] = sub["c"]
        if "d" in sub:
            record["year"] = sub["d"]

    # 215 — физическое описание
    for chunk in fields("215"):
        sub = parse_rusmarc_subfields(chunk)
        if "a" in sub:
            record["quantity"] = sub["a"]
        if "c" in sub:
            record["technique"] = sub["c"]
        if "d" in sub:
            record["dimensions"] = sub["d"]

    # 327 — примечания содержания
    for chunk in fields("327"):
        sub = parse_rusmarc_subfields(chunk)
        if "a" in sub:
            record["content_notes"].append(sub["a"])

    # 540 — варианты заглавия и именованные сущности
    for chunk in fields("540"):
        sub = parse_rusmarc_subfields(chunk)
        if "a" in sub:
            record["variant_titles"].append(sub["a"])

    # 606 — тематические рубрики и даты
    for chunk in fields("606"):
        sub = parse_rusmarc_subfields(chunk)
        if "a" in sub:
            entry = sub["a"]
            if "x" in sub:
                entry += " — " + sub["x"]
            record["topical_subjects"].append(entry)
        if "z" in sub:
            record["topical_subjects_dates"].append(sub["z"])

    # 700 — авторы
    for chunk in fields("700"):
        sub = parse_rusmarc_subfields(chunk)
        parts = []
        if "a" in sub: parts.append(sub["a"])
        if "b" in sub: parts.append(sub["b"])
        full = " ".join(parts) if parts else ""
        if "f" in sub: full += f" ({sub['f']})"
        if full:
            record["authors"].append(full.strip())

    # 852 — шифр хранения
    for chunk in fields("852"):
        sub = parse_rusmarc_subfields(chunk)
        if "b" in sub:
            record["shelf_mark"] = sub["b"]

    # 856 — внешние URL, vivaldi_id
    for chunk in fields("856"):
        sub = parse_rusmarc_subfields(chunk)
        if "u" in sub:
            record["external_urls"].append(sub["u"])
            m = re.search(r"vivaldi\.nlr\.ru/(lo[0-9a-z]+)/", sub["u"])
            if m:
                record["vivaldi_id"] = m.group(1)

    # сформировать единый "reference_ru" — лучшее каталожное описание
    # приоритет: первый content_note → первый variant_title → title
    record["reference_ru"] = (
        record["content_notes"][0] if record["content_notes"]
        else (record["variant_titles"][0] if record["variant_titles"]
              else record["title"])
    )

    return record


# ---------------------------------------------------------------------------
# Основной цикл скрейпинга
# ---------------------------------------------------------------------------

def scrape(args):
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT,
                            "Accept-Language": "ru,en;q=0.8"})

    # этап 1 — перебор catalog_id
    print("=" * 60)
    print("[Phase 1] Enumerating catalog IDs from collection pages")
    print("=" * 60)
    catalog_ids = collect_catalog_ids(session, args.collection_url, delay=args.delay)
    print(f"\n→ Total unique catalog IDs: {len(catalog_ids)}")

    if args.max_items:
        catalog_ids = catalog_ids[: args.max_items]
        print(f"→ Capped to first {len(catalog_ids)} for smoke test")

    # этап 2 — загрузка каждой карточки и миниатюры
    print()
    print("=" * 60)
    print(f"[Phase 2] Fetching {len(catalog_ids)} catalog pages + thumbnails")
    print("=" * 60)

    manifest = []
    n_failed = 0
    n_skipped_image = 0
    t_start = time.time()

    for i, cid in enumerate(catalog_ids):
        catalog_url = f"{BASE}/catalog/{cid}/"
        thumb_url = THUMBNAIL_TEMPLATE.format(catalog_id=cid)
        image_path = images_dir / f"{cid}.jpg"

        # загрузка карточки
        try:
            r = get(session, catalog_url)
            record = parse_catalog_page(r.text, cid)
        except Exception as e:
            print(f"  [{i+1}/{len(catalog_ids)}] FAIL parse {cid}: {e}")
            n_failed += 1
            continue

        # загрузка миниатюры (пропустить, если уже есть)
        if not image_path.exists():
            try:
                ri = get(session, thumb_url, timeout=30)
                ctype = ri.headers.get("Content-Type", "")
                if not ctype.startswith("image/") or len(ri.content) < 1024:
                    print(f"  [{i+1}/{len(catalog_ids)}] {cid}: no image (ctype={ctype}, size={len(ri.content)})")
                    n_skipped_image += 1
                else:
                    image_path.write_bytes(ri.content)
            except Exception as e:
                print(f"  [{i+1}/{len(catalog_ids)}] {cid}: image fetch failed: {e}")
                n_skipped_image += 1

        record["image_path"] = str(image_path) if image_path.exists() else None
        manifest.append(record)

        if (i + 1) % 20 == 0 or i == len(catalog_ids) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(catalog_ids) - (i + 1)) / rate
            print(f"  [{i+1}/{len(catalog_ids)}] rate {rate:.2f} req/sec  "
                  f"ETA {eta/60:.1f} min  ok={len(manifest)} fail={n_failed}  "
                  f"no_img={n_skipped_image}")

        time.sleep(args.delay)

    # сохранить manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "collection_url": args.collection_url,
            "n_catalog_ids": len(catalog_ids),
            "n_records": len(manifest),
            "n_failed": n_failed,
            "n_no_image": n_skipped_image,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": manifest,
        }, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"DONE in {(time.time()-t_start)/60:.1f} min")
    print(f"  Records ok: {len(manifest)}/{len(catalog_ids)}")
    print(f"  Parse failures: {n_failed}")
    print(f"  No image: {n_skipped_image}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Images: {images_dir} ({sum(1 for _ in images_dir.glob('*.jpg'))} files)")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--collection-url", default=COLLECTION_URL)
    p.add_argument("--output-dir", default="data/neb_leningrad_wwii")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Delay between requests in seconds")
    p.add_argument("--max-items", type=int, default=None,
                   help="Cap to N items (for smoke test)")
    return p.parse_args()


if __name__ == "__main__":
    scrape(parse_args())
