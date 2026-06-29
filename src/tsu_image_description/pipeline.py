import logging

from PIL import Image

from .models import CaptionGenerator, Translator
from .metadata_extractor import MetadataExtractor
from .description_builder import DescriptionBuilder
from .caption_cleaner import CaptionCleaner
from .ocr_extractor import OCRExtractor
from .clip_scorer import CLIPScorer
from . import reliability


# Стандартные соотношения сторон (ширина/высота) для бакетирования формата.
_ASPECT_BUCKETS = {
    "1:1": 1.0, "5:4": 1.25, "4:3": 4 / 3, "3:2": 1.5, "16:9": 16 / 9,
    "4:5": 0.8, "3:4": 0.75, "2:3": 2 / 3, "9:16": 9 / 16,
}


def image_format(image_path: str) -> dict:
    """Ориентация и формат открытки из размеров изображения (без ML)."""
    with Image.open(image_path) as im:
        w, h = im.size
    if not h:
        return {"orientation": None, "aspect": None, "width": w, "height": h}
    r = w / h
    if 0.95 <= r <= 1.05:
        orientation = "квадратная"
    elif w > h:
        orientation = "горизонтальная"
    else:
        orientation = "вертикальная"
    aspect = min(_ASPECT_BUCKETS, key=lambda k: abs(_ASPECT_BUCKETS[k] - r))
    return {"orientation": orientation, "aspect": aspect, "width": w, "height": h}


def _init_ocr_with_timeout(ocr_kwargs, timeout=180):
    """Инициализация OCR с таймаутом и мягкой деградацией.

    PaddleOCR требует нативный amd64. Под эмуляцией (amd64-образ на Apple
    Silicon) инициализация зависает молча, без исключения, — обычного
    try/except мало. Запускаем сборку в отдельном потоке; если не уложилась в
    timeout (или упала) — отключаем OCR и продолжаем без него.
    """
    import threading
    box = {}

    def _build():
        try:
            box["ocr"] = OCRExtractor(**ocr_kwargs)
        except Exception as e:  # любая ошибка инициализации не должна валить пайплайн
            box["err"] = e

    t = threading.Thread(target=_build, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        logging.warning(
            "OCR отключён: инициализация не завершилась за %s c "
            "(похоже на paddle под эмуляцией; нужен нативный amd64).", timeout)
        return None
    if "err" in box:
        logging.warning("OCR отключён: %s", box["err"])
        return None
    return box.get("ocr")


class ArchiveDescriptionPipeline:
    def __init__(
        self,
        model_path=None,
        *,
        caption_kwargs=None,
        translator_model=None,
        builder_kwargs=None,
        use_llm_rewriter: bool = False,
        llm_rewriter_kwargs=None,
        taxonomy_version: str = "archival_v2",
        use_ocr: bool = False,
        ocr_kwargs=None,
        use_clipscore: bool = False,
        clipscore_kwargs=None,
    ):
        self.taxonomy_version = taxonomy_version
        self.caption_generator = CaptionGenerator(
            model_path=model_path,
            **(caption_kwargs or {})
        )
        self.translator = (
            Translator(model_name=translator_model) if translator_model
            else Translator()
        )
        self.metadata_extractor = MetadataExtractor(taxonomy_version=taxonomy_version)
        self.caption_cleaner = CaptionCleaner()
        self.description_builder = DescriptionBuilder(**(builder_kwargs or {}))

        # Опциональная стадия OCR. paddleocr — тяжёлая платформозависимая
        # зависимость; при отсутствии/зависании/ошибке — мягкая деградация
        # к пустому OCR-блоку (инициализация под таймаутом, см. helper).
        self.ocr = _init_ocr_with_timeout(ocr_kwargs or {}) if use_ocr else None

        # Опциональный reference-free CLIPScore описания (тот же код, что в
        # офлайн-оценке). Тяжёлый: грузит CLIP + M-CLIP. По умолчанию выключен.
        self.clip_scorer = None
        if use_clipscore:
            try:
                self.clip_scorer = CLIPScorer(**(clipscore_kwargs or {}))
            except ImportError as e:
                logging.warning("CLIPScore отключён: %s", e)

        # Опциональный языковой редактор. Когда включён, заменяет архивное
        # описание от DescriptionBuilder (теги tags_ru всё равно берутся из builder).
        self.llm_rewriter = None
        if use_llm_rewriter:
            from .llm_rewriter import LLMRewriter
            self.llm_rewriter = LLMRewriter(**(llm_rewriter_kwargs or {}))

    def run(self, image_path: str) -> dict:
        caption_en_raw = self.caption_generator.generate(image_path)
        caption_en = self.caption_cleaner.clean_en(caption_en_raw)

        caption_ru_raw = self.translator.translate(caption_en)
        caption_ru = self.caption_cleaner.clean_ru(caption_ru_raw)

        metadata = self.metadata_extractor.extract(image_path)
        inference = MetadataExtractor.infer_theme_mood(metadata)

        # OCR-блок присутствует всегда (пустой при выключенной стадии),
        # чтобы форма ответа была стабильной.
        ocr = self.ocr.extract(image_path) if self.ocr else OCRExtractor.empty_result()

        base_result = {
            "caption": {
                "en": caption_en,
                "ru": caption_ru,
                "ru_raw": caption_ru_raw,
            },
            "metadata": metadata,
            "inference": inference,
            "ocr": ocr,
            "format": image_format(image_path),
        }

        description_result = self.description_builder.build(base_result)

        # Замена шаблонного архивного описания на сгенерированное LLM.
        # tags_ru от builder сохраняются (теги закрытой таксономии).
        if self.llm_rewriter is not None:
            llm_archive = self.llm_rewriter.rewrite(
                caption_en=caption_en,
                metadata=metadata,
                inference=inference,
                # только уверенная надпись доходит до LLM (мусор отсекает гейт)
                ocr_text=ocr["text"] if ocr.get("confident") else None,
            )
            description_result["archive_description_template"] = description_result["archive_description"]
            description_result["archive_description"] = llm_archive

        result = {**base_result, **description_result}

        # Reference-free CLIPScore финального описания (если стадия включена).
        clipscore = None
        if self.clip_scorer is not None:
            clipscore = self.clip_scorer.score(
                image_path, description_result["archive_description"], lang="ru"
            )

        # Сводка надёжности из уже посчитанных сигналов (SigLIP/OCR) + CLIPScore.
        result["reliability"] = reliability.assess(metadata, ocr, clipscore=clipscore)

        return result
