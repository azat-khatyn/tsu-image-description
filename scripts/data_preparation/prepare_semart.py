"""prepare_semart.py — convert SemArt to BLIP-compatible (image, caption) pairs.

SemArt descriptions are art-historical commentary (often 50-200 words). Direct
training would over-bias BLIP toward encyclopedic regime. We extract VISUAL
SENTENCES — short (8-30 words), describing what's IN the painting, not its
historical context.

Filter pipeline per painting:
  1. Split DESCRIPTION into sentences (simple regex).
  2. Filter by word count (8-30).
  3. Drop biographical/context sentences (author names, dates, schools).
  4. Bias toward visual verbs (depicts, shows, stands, holds, wears, ...).
  5. Take up to 2 best sentences per painting → multiple caption rows.

Output:
  data/semart/train.json  list[{id, image_path, caption}]
  data/semart/val.json    same
  data/semart/manifest.json  full + filtering stats
"""

import csv
import json
import random
import re
from pathlib import Path

SEMART_DIR = Path("data/semart/SemArt")
OUTPUT_DIR = Path("data/semart")
IMAGES_DIR = SEMART_DIR / "Images"
SEED = 42

MIN_WORDS = 8
MAX_WORDS = 30
MAX_SENTENCES_PER_PAINTING = 1   # 1 best caption per image (was 2)
MAX_TRAIN_SAMPLES = 8000          # cap to keep M1 training tractable
REQUIRE_VISUAL_CUE = True         # only sentences with explicit visual verbs

# Visual verbs / cues — sentences with these are preferred captions
VISUAL_CUES = [
    r"\bdepicts?\b", r"\bdepicted\b", r"\bdepicting\b",
    r"\bshows?\b", r"\bshowing\b",
    r"\brepresents?\b", r"\brepresented\b", r"\brepresenting\b",
    r"\bportraits?\b", r"\bportrayed\b",
    r"\bpaints?\b", r"\bpainted\b", r"\bpainting\b",
    r"\bdraws?\b", r"\bdrawn\b", r"\bdrawing\b",
    r"\bsits?\b", r"\bsitting\b", r"\bseated\b",
    r"\bstands?\b", r"\bstanding\b",
    r"\bholds?\b", r"\bholding\b",
    r"\bwears?\b", r"\bwearing\b",
    r"\bappears?\b", r"\bappearing\b",
    r"\bvisible\b", r"\bcan be seen\b", r"\bare seen\b",
    r"\bin the (centre|center|background|foreground|distance|middle)\b",
    r"\bon the (left|right|top|bottom)\b",
    r"\bfeatures?\b", r"\bfeaturing\b",
]
VISUAL_RE = re.compile("|".join(VISUAL_CUES), re.IGNORECASE)

# Biographical / contextual patterns to EXCLUDE
EXCLUDE_PATTERNS = [
    r"\bwas (a |an |the )?\w+ painter\b",       # "was a Flemish painter"
    r"\bbelongs? to (the )?\w+ school\b",
    r"\bcommissioned by\b",
    r"\bin the \d{4}s?\b",                       # "in the 1850s"
    r"\bduring (the )?\d{4}",
    r"\b\d{4}\b",                                # ANY 4-digit year (1506, 1850, etc.)
    r"\b(eighteenth|nineteenth|seventeenth|twentieth) century\b",
    r"\bthe artist\b",
    r"\bthe painter\b",
    r"\bwas painted in\b",
    r"\bwas born\b",
    r"\bdied in\b",
    r"\bschool of\b",
    r"\b(was |were )(sent|moved|transferred|destroyed|commissioned)\b",  # historical events
    r"\bover the course of\b",                   # biographical narration
    r"\bthroughout his (career|life)\b",
    r"\bover his (career|life)\b",
]
EXCLUDE_RE = re.compile("|".join(EXCLUDE_PATTERNS), re.IGNORECASE)


def split_sentences(text: str):
    """Simple sentence splitter on .!? followed by whitespace and capital."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Split on terminators followed by space
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def filter_sentence(sentence: str) -> bool:
    """Return True if sentence is a good visual caption candidate."""
    words = sentence.split()
    n = len(words)
    if n < MIN_WORDS or n > MAX_WORDS:
        return False
    if EXCLUDE_RE.search(sentence):
        return False
    # Don't start with author/proper noun chains
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)?\b", sentence):
        # E.g., "Hans Holbein the Younger..." — likely biographical
        # But allow if visual cue present
        if not VISUAL_RE.search(sentence):
            return False
    # Stricter mode: REQUIRE explicit visual cue
    if REQUIRE_VISUAL_CUE and not VISUAL_RE.search(sentence):
        return False
    return True


def rank_sentence(sentence: str) -> float:
    """Score sentence — higher = more visual / useful as caption."""
    n_visual = len(VISUAL_RE.findall(sentence))
    n_words = len(sentence.split())
    # Prefer sentences with visual cues + moderate length (10-20 words)
    length_score = 1.0 if 10 <= n_words <= 20 else 0.6
    return n_visual * 1.5 + length_score


def load_csv(path: Path):
    rows = []
    with open(path, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def build_pairs(rows, label: str):
    """Build (image_path, caption) pairs from SemArt CSV rows."""
    out = []
    stats = {"n_paintings": 0, "n_paintings_kept": 0, "n_sentences_kept": 0,
             "no_image": 0, "no_visual_sentence": 0}

    for r in rows:
        stats["n_paintings"] += 1
        img_file = r.get("IMAGE_FILE", "").strip()
        if not img_file:
            continue

        img_path = IMAGES_DIR / img_file
        if not img_path.exists():
            stats["no_image"] += 1
            continue

        desc = r.get("DESCRIPTION", "")
        sentences = split_sentences(desc)
        candidates = [s for s in sentences if filter_sentence(s)]

        if not candidates:
            stats["no_visual_sentence"] += 1
            continue

        # Sort by rank, take top N
        candidates.sort(key=rank_sentence, reverse=True)
        kept = candidates[:MAX_SENTENCES_PER_PAINTING]

        for sentence in kept:
            out.append({
                "id": f"{label}_{stats['n_paintings']:05d}",
                "image_path": str(img_path),
                "caption": sentence.strip().rstrip("."),
                "technique": r.get("TECHNIQUE", "").strip(),
                "type": r.get("TYPE", "").strip(),
                "timeframe": r.get("TIMEFRAME", "").strip(),
            })

        stats["n_paintings_kept"] += 1
        stats["n_sentences_kept"] += len(kept)

    return out, stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading SemArt CSV files")
    train_rows = load_csv(SEMART_DIR / "semart_train.csv")
    val_rows = load_csv(SEMART_DIR / "semart_val.csv")
    print(f"      Train: {len(train_rows)} paintings")
    print(f"      Val:   {len(val_rows)} paintings")
    print()

    print("[2/4] Extracting visual sentences (filter pipeline)")
    train_pairs, train_stats = build_pairs(train_rows, "train")
    val_pairs, val_stats = build_pairs(val_rows, "val")
    print(f"      Train stats: {train_stats}")
    print(f"      Val stats:   {val_stats}")
    print()

    # Cap train size to MAX_TRAIN_SAMPLES (random subsample if needed)
    if len(train_pairs) > MAX_TRAIN_SAMPLES:
        random.seed(SEED)
        random.shuffle(train_pairs)
        train_pairs = train_pairs[:MAX_TRAIN_SAMPLES]
        print(f"      Train capped to {MAX_TRAIN_SAMPLES} (random sample, seed={SEED})")

    print(f"[3/4] Saving {len(train_pairs)} train + {len(val_pairs)} val pairs")
    with open(OUTPUT_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_pairs, f, indent=2, ensure_ascii=False)
    print(f"      Train: {OUTPUT_DIR / 'train.json'}")
    print(f"      Val:   {OUTPUT_DIR / 'val.json'}")
    print()

    print("[4/4] Sample captions:")
    random.seed(SEED)
    sampled = random.sample(train_pairs, min(10, len(train_pairs)))
    for s in sampled:
        print(f"   • [{s['type']:>10}] {s['caption'][:120]}")
    print()

    # Word count distribution
    import statistics
    wc = [len(p["caption"].split()) for p in train_pairs]
    print(f"Caption word count: min={min(wc)} median={statistics.median(wc):.0f} "
          f"mean={statistics.mean(wc):.1f} max={max(wc)}")

    print("\nDONE. Ready for BLIP LoRA training.")


if __name__ == "__main__":
    main()
