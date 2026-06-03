from typing import Dict, List

from PIL import Image
import torch
from transformers import (
    AutoModel,
    AutoProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class CaptionGenerator:
    """Генератор подписей с бэкендами BLIP-1 / BLIP-2.

    backend:
      - "blip1" (по умолчанию)
      - "blip2"

    Примечания:
      - BLIP-2 повышает качество подписи, но тяжелее на Apple M1.
      - BLIP-1 — безопасный дефолт; BLIP-2 включается экспериментально.
    """

    def __init__(
        self,
        model_path=None,
        *,
        backend: str = "blip1",
        num_beams: int = 1,
        length_penalty: float = 1.0,
        prompt_prefix: str | None = None,
        max_new_tokens: int = 50,
    ):
        self.backend = backend
        self.device = get_device()
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.prompt_prefix = (prompt_prefix or "").strip() or None
        self.max_new_tokens = max_new_tokens

        if self.backend == "blip2":
            self.model_path = model_path or "Salesforce/blip2-opt-2.7b"
            self.processor = Blip2Processor.from_pretrained(self.model_path)

            dtype = torch.float16 if self.device != "cpu" else torch.float32
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
            ).to(self.device)
        else:
            # Дефолтный бэкенд BLIP-1: модель переключена с base на large.
            self.model_path = model_path or "Salesforce/blip-image-captioning-large"

            # Распознавание PEFT (LoRA) адаптера: папка с adapter_config.json
            # без полного model.safetensors. Грузим базовую модель из конфига
            # адаптера и надеваем адаптер поверх.
            from pathlib import Path
            adapter_cfg = Path(self.model_path) / "adapter_config.json"
            is_adapter = adapter_cfg.is_file()

            if is_adapter:
                import json as _json
                from peft import PeftModel
                with open(adapter_cfg) as _f:
                    cfg = _json.load(_f)
                base_name = cfg.get("base_model_name_or_path", "Salesforce/blip-image-captioning-large")
                print(f"[CaptionGenerator] Detected PEFT adapter at {self.model_path}; base={base_name}")
                self.processor = BlipProcessor.from_pretrained(self.model_path)
                base = BlipForConditionalGeneration.from_pretrained(base_name)
                self.model = PeftModel.from_pretrained(base, self.model_path).to(self.device)
                self.model.eval()
            else:
                self.processor = BlipProcessor.from_pretrained(self.model_path)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_path
                ).to(self.device)

        print(
            f"[CaptionGenerator] backend={self.backend} model={self.model_path} "
            f"(num_beams={self.num_beams}, length_penalty={self.length_penalty}, "
            f"prompt_prefix={self.prompt_prefix!r})"
            )

    def generate(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")

        if self.prompt_prefix:
            inputs = self.processor(
                images=image,
                text=self.prompt_prefix,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                do_sample=False,
            )

        caption = self.processor.decode(output[0], skip_special_tokens=True).strip()

        if self.prompt_prefix:
            lc_caption = caption.lower()
            lc_prefix = self.prompt_prefix.lower()
            if lc_caption.startswith(lc_prefix):
                caption = caption[len(self.prompt_prefix):].lstrip(" ,.;:")

        return caption


class Translator:
    """Переводчик EN→RU. Поддерживает бэкенды MarianMT и NLLB-200."""

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-ru"):
        self.device = get_device()
        self.model_name = model_name
        self.is_nllb = "nllb" in model_name.lower()

        print(
            f"[Translator] Using model: {model_name} "
            f"(backend: {'NLLB-200' if self.is_nllb else 'MarianMT'})"
        )

        if self.is_nllb:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids("rus_Cyrl")
        else:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
            self.tgt_lang_id = None

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        if self.is_nllb:
            translated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tgt_lang_id,
                max_length=256,
            )
        else:
            translated = self.model.generate(**inputs)

        return self.tokenizer.decode(translated[0], skip_special_tokens=True)


class SigLIPZeroShotClassifier:
    """SigLIP zero-shot скоринг: (изображение, тексты-кандидаты) -> {label: prob}.

    Тонкая обёртка над моделью, без гейтинга и таксономии (они — в
    metadata_extractor.py). Свап SigLIP-1 -> SigLIP-2 затрагивает только этот класс.
    """

    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        self.model_name = model_name
        self.device = get_device()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def score(self, image: Image.Image, candidates: List[str]) -> Dict[str, float]:
        inputs = self.processor(
            text=candidates,
            images=image,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

        return {candidate: float(score) for candidate, score in zip(candidates, probs)}
