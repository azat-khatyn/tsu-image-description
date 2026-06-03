"""CLIPScorer — reference-free CLIPScore (косинус картинка↔текст).

Та же модель и формула, что в офлайн-оценке: картинку кодирует open_clip
ViT-B-32 (openai), RU-текст — M-CLIP (XLM-R + линейная проекция), EN-текст —
текстовый энкодер того же ViT-B-32; CLIPScore = косинус L2-нормированных
эмбеддингов. scripts/evaluate.py импортирует этот класс, поэтому рантайм-сигнал
и метрика ВКР считаются ОДНИМ кодом (не расходятся).

open_clip/transformers — тяжёлые зависимости; open_clip импортируется лениво
(в __init__), как у OCRExtractor: без библиотеки модуль импортируется, ошибка —
только при создании экземпляра, пайплайн мягко деградирует (clipscore=None).
"""

import json
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .models import get_device

DEFAULT_MCLIP_MODEL = "M-CLIP/XLM-Roberta-Large-Vit-B-32"


class _MCLIPTextEncoder(torch.nn.Module):
    """Текстовый энкодер M-CLIP: XLM-R + линейная проекция."""

    def __init__(self, transformer, linear):
        super().__init__()
        self.transformer = transformer
        self.LinearTransformation = linear


class CLIPScorer:
    """CLIPScore(image, text) = косинус L2-нормированных эмбеддингов.

    Args:
        mclip_model: HF-идентификатор M-CLIP текстового энкодера (RU).
        device: устройство; по умолчанию get_device().
        load_ru: грузить M-CLIP (для русского текста).
    """

    def __init__(
        self,
        *,
        mclip_model: str = DEFAULT_MCLIP_MODEL,
        device: Optional[str] = None,
        load_ru: bool = True,
    ):
        self.device = device or get_device()
        self.mclip_name = mclip_model

        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "CLIPScorer требует open_clip_torch (есть в requirements.txt)."
            ) from e

        # EN CLIP: кодирует картинку всегда; его текстовый энкодер — для en-текста.
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.clip_en = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer_en = open_clip.get_tokenizer("ViT-B-32")

        # M-CLIP: текстовый энкодер для русского.
        self.mclip_text = None
        self.mclip_tokenizer = None
        if load_ru:
            self.mclip_text, self.mclip_tokenizer = self._load_mclip(mclip_model, self.device)

    @staticmethod
    def _load_mclip(model_name, device):
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as cf:
            cfg = json.load(cf)

        base_model_name = cfg.get("modelBase", "xlm-roberta-large")
        transformer_dim = cfg.get("transformerDimensions", 1024)
        num_dims = cfg.get("numDims", 512)

        try:
            weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
            from safetensors.torch import load_file

            state = load_file(weights_path)
        except Exception:
            weights_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
            state = torch.load(weights_path, map_location="cpu")

        base_cfg = AutoConfig.from_pretrained(base_model_name)
        transformer = AutoModel.from_config(base_cfg)

        transformer_state = {
            k[len("transformer."):]: v
            for k, v in state.items()
            if k.startswith("transformer.")
        }
        missing, unexpected = transformer.load_state_dict(transformer_state, strict=False)
        if missing or unexpected:
            print(
                f"[CLIPScorer] M-CLIP transformer load: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys"
            )

        linear = torch.nn.Linear(transformer_dim, num_dims)
        linear.load_state_dict(
            {
                "weight": state["LinearTransformation.weight"],
                "bias": state["LinearTransformation.bias"],
            }
        )

        text_model = _MCLIPTextEncoder(transformer, linear).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        return text_model, tokenizer

    @torch.no_grad()
    def encode_image(self, image_path) -> np.ndarray:
        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        feat = self.clip_en.encode_image(image)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text: str, lang: str = "ru") -> np.ndarray:
        """Эмбеддинг текста: lang='en' — энкодер ViT-B-32, иначе M-CLIP (RU)."""
        if lang == "en":
            return self._encode_text_en(text)
        return self._encode_text_ru(text)

    @torch.no_grad()
    def _encode_text_en(self, text) -> np.ndarray:
        tokens = self.tokenizer_en([text]).to(self.device)
        feat = self.clip_en.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def _encode_text_ru(self, text) -> np.ndarray:
        if self.mclip_text is None:
            raise RuntimeError("M-CLIP не загружен (создан с load_ru=False).")
        tok = self.mclip_tokenizer([text], padding=True, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}

        embs = self.mclip_text.transformer(**tok)[0]
        att = tok["attention_mask"]
        pooled = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        feat = self.mclip_text.LinearTransformation(pooled)

        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    @staticmethod
    def cosine(a, b) -> float:
        return float(np.dot(a, b))

    def score(self, image_path, text: str, lang: str = "ru") -> float:
        """CLIPScore для одной пары (картинка, текст): косинус эмбеддингов."""
        return self.cosine(self.encode_image(image_path), self.encode_text(text, lang))
