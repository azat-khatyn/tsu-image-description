"""exp_db.py — трекер экспериментов на SQLite.

Стандартизирует хранение и сравнение результатов экспериментов.

Схема:
  experiments — одна строка на (config, pool)
  per_item    — на каждое изображение: caption_en, caption_ru, archive_ru, sds_axes, retrieval_rank
  metrics     — агрегаты (CLIPScore, R@k, SDS) с разбиением по source

Использование:
  # инициализация БД
  python scripts/exp_db.py init

  # загрузка одного JSON
  python scripts/exp_db.py ingest \\
      --json data/eval/results/final/metrics_proposed_display_n220.json \\
      --exp-id E00_proposed_display_n220 \\
      --block A \\
      --name "Proposed display (BLIP-large + Marian + full -theme -mood)"

  # дополнительно прикрепить SDS-файл
  python scripts/exp_db.py ingest-sds \\
      --json data/eval/results/final/sds_proposed_display_n220.json \\
      --exp-id E00_proposed_display_n220

  # список экспериментов
  python scripts/exp_db.py list

  # сравнение двух экспериментов
  python scripts/exp_db.py compare --exp-a E00_v1 --exp-b E00_v2

  # вывод метрик
  python scripts/exp_db.py metrics --exp-id E00_proposed_display_n220
"""

import argparse
import json
import os
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

# корень проекта вычисляется относительно этого файла (scripts/exp_db.py).
# абсолютный путь, чтобы БД находилась независимо от cwd (например, из Jupyter в notebooks/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "eval" / "experiments.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    exp_id       TEXT UNIQUE NOT NULL,
    block        TEXT,
    name         TEXT,
    config_json  TEXT NOT NULL,
    git_commit   TEXT,
    timestamp    TEXT NOT NULL,
    n_items      INTEGER,
    note         TEXT,
    source_file  TEXT
);

CREATE TABLE IF NOT EXISTS per_item (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id       INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    image_path          TEXT,
    source              TEXT,
    caption_en          TEXT,
    caption_ru          TEXT,
    caption_ru_raw      TEXT,
    archive_ru          TEXT,
    reference_ru        TEXT,
    sds_axes_json       TEXT,
    sds_value           REAL,
    retrieval_rank_i2t  INTEGER,
    retrieval_rank_t2i  INTEGER,
    latency_sec         REAL,
    scores_json         TEXT,
    extra_json          TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    metric_name   TEXT NOT NULL,
    source        TEXT,
    n             INTEGER,
    value         REAL,
    ci_low        REAL,
    ci_high       REAL
);

CREATE INDEX IF NOT EXISTS idx_experiments_exp_id ON experiments(exp_id);
CREATE INDEX IF NOT EXISTS idx_per_item_experiment ON per_item(experiment_id);
CREATE INDEX IF NOT EXISTS idx_per_item_image      ON per_item(image_path);
CREATE INDEX IF NOT EXISTS idx_metrics_experiment  ON metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name        ON metrics(metric_name);
"""


# ---------------------------------------------------------------------------
# Работа с БД
# ---------------------------------------------------------------------------

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"[INFO] Initialized DB at {DB_PATH}")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Загрузка metrics_triad_*.json (из evaluate.py)
# ---------------------------------------------------------------------------

def ingest_triad(
    json_path: str,
    *,
    exp_id: str,
    block: str | None = None,
    name: str | None = None,
    note: str | None = None,
    replace: bool = False,
):
    payload = json.load(open(json_path, "r", encoding="utf-8"))
    summary = payload["summary"]
    per_item = payload.get("per_item", [])

    conn = get_conn()

    # если запись уже есть — при флаге replace перезаписываем
    existing = conn.execute(
        "SELECT id FROM experiments WHERE exp_id = ?", (exp_id,)
    ).fetchone()
    if existing:
        if not replace:
            print(f"[WARN] exp_id={exp_id} already exists. Use --replace to overwrite.")
            conn.close()
            return existing["id"]
        # удаление каскадом затрагивает per_item / metrics
        conn.execute("DELETE FROM experiments WHERE exp_id = ?", (exp_id,))
        print(f"[INFO] Replaced existing exp_id={exp_id}")

    cur = conn.execute(
        """INSERT INTO experiments
           (exp_id, block, name, config_json, git_commit, timestamp, n_items, note, source_file)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            exp_id,
            block,
            name,
            json.dumps(summary.get("config", {}), ensure_ascii=False),
            git_commit(),
            datetime.now().isoformat(timespec="seconds"),
            summary.get("num_examples", len(per_item)),
            note,
            str(json_path),
        ),
    )
    experiment_id = cur.lastrowid

    # построчная запись per_item
    for it in per_item:
        retrieval = it.get("retrieval", {}) or {}
        conn.execute(
            """INSERT INTO per_item
               (experiment_id, image_path, source, caption_en, caption_ru, caption_ru_raw,
                archive_ru, reference_ru, sds_axes_json, sds_value,
                retrieval_rank_i2t, retrieval_rank_t2i, latency_sec,
                scores_json, extra_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                it.get("image_path"),
                it.get("source"),
                it.get("caption_en"),
                it.get("caption_ru"),
                it.get("caption_ru_raw"),
                it.get("archive_ru"),
                it.get("reference_ru"),
                retrieval.get("rank_i2t"),
                retrieval.get("rank_t2i"),
                it.get("latency_sec"),
                json.dumps(it.get("scores", {}), ensure_ascii=False),
                json.dumps(
                    {k: v for k, v in it.items() if k not in {
                        "image_path", "source", "caption_en", "caption_ru",
                        "caption_ru_raw", "archive_ru", "reference_ru",
                        "scores", "retrieval", "latency_sec",
                    }},
                    ensure_ascii=False,
                ),
            ),
        )

    # агрегатные метрики — плоский набор ключей + варианты по source
    def _log_metric(name_, source_, n_, value_):
        conn.execute(
            "INSERT INTO metrics (experiment_id, metric_name, source, n, value) VALUES (?, ?, ?, ?, ?)",
            (experiment_id, name_, source_, n_, value_),
        )

    # общие метрики
    metrics_block = summary.get("metrics", {})
    for k, v in metrics_block.items():
        if v is not None:
            _log_metric(k, "all", summary.get("num_examples"), v)

    # retrieval
    retrieval = summary.get("retrieval", {})
    n_pool = retrieval.get("n_pool", summary.get("num_examples"))
    for direction in ("i2t", "t2i"):
        d = retrieval.get(direction, {})
        for k_, v_ in d.items():
            _log_metric(f"{direction}_{k_}", "all", n_pool, v_)
    if "mean_rank_i2t" in retrieval:
        _log_metric("mean_rank_i2t", "all", n_pool, retrieval["mean_rank_i2t"])
    if "mean_rank_t2i" in retrieval:
        _log_metric("mean_rank_t2i", "all", n_pool, retrieval["mean_rank_t2i"])

    # задержка
    latency = summary.get("latency", {})
    if latency.get("mean_sec") is not None:
        _log_metric("latency_mean_sec", "all", summary.get("num_examples"), latency["mean_sec"])
    if latency.get("images_per_sec") is not None:
        _log_metric("images_per_sec", "all", summary.get("num_examples"), latency["images_per_sec"])

    # разбиение по source
    by_source = summary.get("by_source", {})
    for src, src_block in by_source.items():
        n_src = src_block.get("n")
        for k, v in src_block.items():
            if k == "n":
                continue
            if v is not None:
                _log_metric(k, src, n_src, v)

    conn.commit()
    conn.close()
    print(f"[INFO] Ingested triad: exp_id={exp_id} (id={experiment_id}, n={len(per_item)})")
    return experiment_id


# ---------------------------------------------------------------------------
# Загрузка sds_*.json (из compute_sds.py)
# ---------------------------------------------------------------------------

def ingest_sds(
    json_path: str,
    *,
    exp_id: str,
):
    payload = json.load(open(json_path, "r", encoding="utf-8"))
    aggregates = payload.get("aggregates", {})
    per_item = payload.get("per_item", [])

    conn = get_conn()
    row = conn.execute("SELECT id FROM experiments WHERE exp_id = ?", (exp_id,)).fetchone()
    if not row:
        print(f"[ERROR] exp_id={exp_id} not found. Ingest triad first.")
        conn.close()
        return None
    experiment_id = row["id"]

    # обновляем строки per_item: проставляем sds_axes_json + sds_value
    for it in per_item:
        sds = it.get("sds", {})
        sds_value = sds.get("sds")
        sds_axes = {k: v for k, v in sds.items() if k != "sds"}
        conn.execute(
            "UPDATE per_item SET sds_axes_json = ?, sds_value = ? "
            "WHERE experiment_id = ? AND image_path = ?",
            (
                json.dumps(sds_axes, ensure_ascii=False),
                sds_value,
                experiment_id,
                it.get("image_path"),
            ),
        )

    # агрегатные метрики SDS
    def _log_metric(name_, source_, n_, value_):
        # идемпотентно: при наличии записи перезаписываем
        conn.execute(
            "DELETE FROM metrics WHERE experiment_id = ? AND metric_name = ? AND source = ?",
            (experiment_id, name_, source_),
        )
        conn.execute(
            "INSERT INTO metrics (experiment_id, metric_name, source, n, value) VALUES (?, ?, ?, ?, ?)",
            (experiment_id, name_, source_, n_, value_),
        )

    if aggregates.get("mean_sds") is not None:
        _log_metric("SDS_mean", "all", aggregates.get("n"), aggregates["mean_sds"])

    # покрытие по каждой оси — отдельными метриками
    for axis, value in aggregates.get("per_axis_coverage", {}).items():
        _log_metric(f"SDS_axis_{axis}", "all", aggregates.get("n"), value)

    # разбиение по source
    for src, src_block in aggregates.get("by_source", {}).items():
        _log_metric("SDS_mean", src, src_block.get("n"), src_block.get("mean_sds"))

    conn.commit()
    conn.close()
    print(f"[INFO] Patched SDS for exp_id={exp_id}")


# ---------------------------------------------------------------------------
# Команды запросов
# ---------------------------------------------------------------------------

def cmd_list(args):
    conn = get_conn()
    rows = conn.execute(
        """SELECT e.exp_id, e.block, e.name, e.n_items, e.timestamp,
                  (SELECT value FROM metrics m
                   WHERE m.experiment_id=e.id AND m.metric_name='CLIPScore_RU_archive_ru' AND m.source='all')
                   AS clipscore,
                  (SELECT value FROM metrics m
                   WHERE m.experiment_id=e.id AND m.metric_name='t2i_R@1' AND m.source='all')
                   AS r1,
                  (SELECT value FROM metrics m
                   WHERE m.experiment_id=e.id AND m.metric_name='t2i_R@10' AND m.source='all')
                   AS r10,
                  (SELECT value FROM metrics m
                   WHERE m.experiment_id=e.id AND m.metric_name='SDS_mean' AND m.source='all')
                   AS sds
           FROM experiments e
           ORDER BY e.timestamp""",
    ).fetchall()
    conn.close()

    if not rows:
        print("[INFO] No experiments yet.")
        return

    print(f"\n{'exp_id':<40} {'blk':<3} {'n':>5} {'CLIP':>7} {'R@1':>6} {'R@10':>6} {'SDS':>6}  name")
    print("-" * 130)
    for r in rows:
        clip = f"{r['clipscore']:.4f}" if r['clipscore'] is not None else "  -  "
        r1 = f"{r['r1']:.3f}" if r['r1'] is not None else "  -  "
        r10 = f"{r['r10']:.3f}" if r['r10'] is not None else "  -  "
        sds = f"{r['sds']:.3f}" if r['sds'] is not None else "  -  "
        n = str(r['n_items']) if r['n_items'] else "-"
        blk = r['block'] or "-"
        name = (r['name'] or "")[:60]
        print(f"{r['exp_id']:<40} {blk:<3} {n:>5} {clip:>7} {r1:>6} {r10:>6} {sds:>6}  {name}")


def cmd_metrics(args):
    conn = get_conn()
    rows = conn.execute(
        """SELECT m.metric_name, m.source, m.n, m.value
           FROM metrics m
           JOIN experiments e ON e.id = m.experiment_id
           WHERE e.exp_id = ?
           ORDER BY m.source, m.metric_name""",
        (args.exp_id,),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"[INFO] No metrics for exp_id={args.exp_id}")
        return

    print(f"\nMetrics for {args.exp_id}:")
    print(f"{'metric':<32} {'source':<10} {'n':>5} {'value':>10}")
    print("-" * 65)
    for r in rows:
        n = str(r["n"]) if r["n"] else "-"
        v = f"{r['value']:.4f}" if r['value'] is not None else "n/a"
        print(f"{r['metric_name']:<32} {r['source']:<10} {n:>5} {v:>10}")


def cmd_compare(args):
    conn = get_conn()
    rows = conn.execute(
        """SELECT
              m.metric_name,
              m.source,
              MAX(CASE WHEN e.exp_id = ? THEN m.value END) AS val_a,
              MAX(CASE WHEN e.exp_id = ? THEN m.value END) AS val_b
           FROM metrics m
           JOIN experiments e ON e.id = m.experiment_id
           WHERE e.exp_id IN (?, ?)
           GROUP BY m.metric_name, m.source
           ORDER BY m.source, m.metric_name""",
        (args.exp_a, args.exp_b, args.exp_a, args.exp_b),
    ).fetchall()
    conn.close()

    print(f"\nCompare: {args.exp_a}  vs  {args.exp_b}")
    print(f"{'metric':<32} {'source':<10} {'A':>10} {'B':>10} {'Δ (B-A)':>10}")
    print("-" * 80)
    for r in rows:
        a = r["val_a"]
        b = r["val_b"]
        if a is None or b is None:
            continue
        d = b - a
        print(f"{r['metric_name']:<32} {r['source']:<10} {a:>10.4f} {b:>10.4f} {d:>+10.4f}")


def cmd_show(args):
    """Выводит все сгенерированные описания для одного эксперимента."""
    conn = get_conn()
    where = "e.exp_id = ?"
    params = [args.exp_id]
    if args.source and args.source != "all":
        where += " AND p.source = ?"
        params.append(args.source)

    rows = conn.execute(
        f"""SELECT p.image_path, p.source, p.reference_ru, p.caption_en,
                   p.caption_ru, p.caption_ru_raw, p.archive_ru, p.sds_value,
                   p.retrieval_rank_t2i
           FROM per_item p
           JOIN experiments e ON e.id = p.experiment_id
           WHERE {where}
           ORDER BY p.source, p.image_path
           {f'LIMIT {int(args.limit)}' if args.limit else ''}""",
        params,
    ).fetchall()
    conn.close()

    if not rows:
        print(f"[INFO] No items for exp_id={args.exp_id}")
        return

    if args.format == "markdown":
        print(f"# Descriptions: {args.exp_id}\n")
        print("| # | image | source | reference | archive_description | SDS | rank |")
        print("|---|---|---|---|---|---|---|")
        for i, r in enumerate(rows, 1):
            img_name = Path(r["image_path"]).name
            ref = (r["reference_ru"] or "").replace("|", "\\|")[:60]
            arc = (r["archive_ru"] or "").replace("|", "\\|")[:140]
            sds = f"{r['sds_value']:.2f}" if r['sds_value'] is not None else "-"
            rk = str(r["retrieval_rank_t2i"]) if r["retrieval_rank_t2i"] is not None else "-"
            print(f"| {i} | {img_name} | {r['source']} | {ref} | {arc} | {sds} | {rk} |")
        return

    if args.format == "jsonl":
        for r in rows:
            print(json.dumps({k: r[k] for k in r.keys()}, ensure_ascii=False))
        return

    # по умолчанию — таблица для чтения человеком
    print(f"\nDescriptions for {args.exp_id}  (n={len(rows)})")
    print("=" * 110)
    for i, r in enumerate(rows, 1):
        img_name = Path(r["image_path"]).name
        extras = []
        if r["retrieval_rank_t2i"] is not None:
            extras.append(f"rank_t2i={r['retrieval_rank_t2i']}")
        if r["sds_value"] is not None:
            extras.append(f"SDS={r['sds_value']:.2f}")
        extra_str = ("  " + "  ".join(extras)) if extras else ""
        print(f"\n{i}. {img_name}  [{r['source']}]{extra_str}")
        if r["reference_ru"]:
            print(f"   REF:        {r['reference_ru']}")
        if args.show_intermediate:
            print(f"   caption_en: {r['caption_en']}")
            print(f"   caption_ru: {r['caption_ru']}")
            if r["caption_ru_raw"] and r["caption_ru_raw"] != r["caption_ru"]:
                print(f"   ru_raw:     {r['caption_ru_raw']}")
        print(f"   archive:    {r['archive_ru']}")


def cmd_per_item(args):
    """Сравнивает один image_path в нескольких экспериментах бок о бок."""
    conn = get_conn()
    exp_ids = args.exp_ids
    placeholders = ",".join("?" * len(exp_ids))
    rows = conn.execute(
        f"""SELECT e.exp_id, p.image_path, p.caption_en, p.caption_ru, p.archive_ru,
                   p.sds_value, p.retrieval_rank_t2i
           FROM per_item p
           JOIN experiments e ON e.id = p.experiment_id
           WHERE e.exp_id IN ({placeholders})
             AND p.image_path = ?
           ORDER BY e.timestamp""",
        (*exp_ids, args.image_path),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"[INFO] No items for {args.image_path}")
        return

    print(f"\nimage: {args.image_path}")
    for r in rows:
        print(f"\n  exp_id: {r['exp_id']}")
        print(f"  caption_en: {r['caption_en']}")
        print(f"  caption_ru: {r['caption_ru']}")
        print(f"  archive:    {r['archive_ru']}")
        if r["sds_value"] is not None:
            print(f"  SDS:        {r['sds_value']:.3f}")
        if r["retrieval_rank_t2i"] is not None:
            print(f"  rank_t2i:   {r['retrieval_rank_t2i']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--json", required=True)
    p_ing.add_argument("--exp-id", required=True)
    p_ing.add_argument("--block", default=None)
    p_ing.add_argument("--name", default=None)
    p_ing.add_argument("--note", default=None)
    p_ing.add_argument("--replace", action="store_true")

    p_sds = sub.add_parser("ingest-sds")
    p_sds.add_argument("--json", required=True)
    p_sds.add_argument("--exp-id", required=True)

    sub.add_parser("list")

    p_met = sub.add_parser("metrics")
    p_met.add_argument("--exp-id", required=True)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--exp-a", required=True)
    p_cmp.add_argument("--exp-b", required=True)

    p_show = sub.add_parser("show", help="Show generated descriptions for one experiment")
    p_show.add_argument("--exp-id", required=True)
    p_show.add_argument("--source", default="all", choices=["all", "rgb", "nypl"])
    p_show.add_argument("--format", default="table", choices=["table", "markdown", "jsonl"])
    p_show.add_argument("--limit", type=int, default=None)
    p_show.add_argument("--show-intermediate", action="store_true",
                        help="Also print caption_en and caption_ru (default: only archive)")

    p_pi = sub.add_parser("per-item")
    p_pi.add_argument("--exp-ids", nargs="+", required=True)
    p_pi.add_argument("--image-path", required=True)

    args = parser.parse_args()

    if args.cmd == "init":
        init_db()
    elif args.cmd == "ingest":
        ingest_triad(
            args.json,
            exp_id=args.exp_id,
            block=args.block,
            name=args.name,
            note=args.note,
            replace=args.replace,
        )
    elif args.cmd == "ingest-sds":
        ingest_sds(args.json, exp_id=args.exp_id)
    elif args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "metrics":
        cmd_metrics(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    elif args.cmd == "show":
        cmd_show(args)
    elif args.cmd == "per-item":
        cmd_per_item(args)


if __name__ == "__main__":
    main()
