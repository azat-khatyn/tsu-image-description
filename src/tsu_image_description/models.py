from PIL import Image
import torch
from transformers import (
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
    """Unified caption generator with BLIP-1 / BLIP-2 backends.

    backend:
      - "blip1" (default)
      - "blip2"

    Notes:
      - BLIP-2 can improve caption quality, but may be heavy on Apple M1.
      - Keep BLIP-1 as safe default and switch to BLIP-2 experimentally.
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
            # Default BLIP-1 backend switched from base to large.
            self.model_path = model_path or "Salesforce/blip-image-captioning-large"

            # Detect PEFT (LoRA) adapter: folder containing adapter_config.json
            # without a full model.safetensors. Load base from adapter_config and
            # attach the adapter on top.
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
    """EN→RU translator. Supports both MarianMT and NLLB-200."""

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
