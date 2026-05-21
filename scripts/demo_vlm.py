"""demo_vlm.py — quick demo: run Qwen2-VL-2B-Instruct on a few RGB postcards
with an archive-style prompt to assess output quality.

Not a full evaluation — just produces text for visual inspection.
Use this to decide if VLM-based pipeline is worth pursuing as the substantive method.

    python scripts/demo_vlm.py --n 10
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# Prompt variant 1 — explicit "library cataloguer" with structured guidance.
SYSTEM_PROMPT = (
    "Ты — каталогизатор Российской государственной библиотеки. "
    "Опиши изображённую открытку для архивного поиска. "
    "Включи в описание:\n"
    "1. Тип материала (открытка / плакат / иллюстрация / фотография).\n"
    "2. Визуальный сюжет — что изображено, кто, где.\n"
    "3. Стиль (винтажная иллюстрация / гравюра / черно-белая фотография / живопись).\n"
    "4. Эпоху, если можно определить.\n"
    "5. Культурно-исторический контекст, только если он очевиден.\n\n"
    "Стиль изложения: каталожный, нейтральный, описательный. Без оценочных слов "
    "('красивая', 'прекрасная'). 2-3 предложения. Только то, что реально видно. "
    "Не выдумывай детали."
)

USER_PROMPT = "Опиши эту открытку для архивного каталога."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="how many RGB postcards to process")
    p.add_argument("--references", type=str, default="data/eval/references.jsonl")
    p.add_argument("--output", type=str, default="data/eval/results/demo_qwen2vl.jsonl")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps"],
        help="cpu is slow (~30s/img) but stable; mps is faster but may swap heavily",
    )
    return p.parse_args()


def load_rgb_references(path: str, n: int):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("source", "rgb") == "rgb":
                items.append(item)
            if len(items) >= n:
                break
    return items


def main():
    args = parse_args()

    device = args.device
    # bfloat16 on CPU is faster than float32 and avoids float16-on-CPU compatibility issues
    dtype = torch.bfloat16 if device == "cpu" else torch.float16

    print(f"[INFO] Device: {device}  dtype: {dtype}", flush=True)

    refs = load_rgb_references(args.references, args.n)
    print(f"[INFO] Loaded {len(refs)} RGB postcards", flush=True)

    print(f"[INFO] Loading {MODEL_NAME} (first run downloads ~5GB)...", flush=True)
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print(f"[INFO] Model loaded in {time.time() - t0:.1f}s", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Open file in append-line mode; write after each generation for live progress.
    with open(out_path, "w", encoding="utf-8") as f_out:
        for i, ref in enumerate(refs):
            image_path = ref["image_path"]
            reference = ref.get("reference_short_ru", "")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            t0 = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False
                )
            elapsed = time.time() - t0

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            print(f"\n[{i+1}/{len(refs)}] {Path(image_path).name}  ({elapsed:.1f}s)", flush=True)
            print(f"  reference: {reference}", flush=True)
            print(f"  Qwen2-VL:  {output_text}", flush=True)

            entry = {
                "image_path": image_path,
                "reference_short_ru": reference,
                "qwen2vl_description": output_text,
                "elapsed_sec": elapsed,
                "prompt_variant": 1,
                "device": device,
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"\n[INFO] Saved descriptions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
