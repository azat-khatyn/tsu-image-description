from functools import lru_cache
from threading import Lock
import logging

from app.core.config import settings
from tsu_image_description.models import get_device
from tsu_image_description.pipeline import ArchiveDescriptionPipeline


class InferenceService:
    """Сервисная обёртка над `ArchiveDescriptionPipeline`.

    Конфигурация пайплайна полностью берётся из `app.core.config.settings`
    (которая, в свою очередь, читает переменные окружения).
    Значения по умолчанию соответствуют «предложенному решению» из README:
    BLIP-large + SigLIP с каталожной таксономией + языковой редактор
    Vikhr-Nemo-12B с архивной prompt-инструкцией.
    """

    def __init__(self) -> None:
        self._pipeline: ArchiveDescriptionPipeline | None = None
        self._lock = Lock()
        self.device = get_device()

        # Снимок конфигурации для эндпоинта /health и интерфейса.
        # Создаётся сразу, чтобы UI мог показать архитектуру до первого инференса.
        self.pipeline_config: dict = {
            "caption_model": settings.caption_model,
            "caption_backend": settings.caption_backend,
            "taxonomy_version": settings.taxonomy_version,
            "use_llm_rewriter": settings.use_llm_rewriter,
            "llm_prompt_style": (
                settings.llm_prompt_style if settings.use_llm_rewriter else None
            ),
            "llm_model": settings.llm_model if settings.use_llm_rewriter else None,
            "use_ocr": settings.use_ocr,
            "use_clipscore": settings.use_clipscore,
        }

    @property
    def model_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def ocr_available(self) -> bool | None:
        """OCR реально поднялся? None — пайплайн ещё не загружен."""
        if self._pipeline is None:
            return None
        return self._pipeline.ocr is not None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    llm_kwargs = (
                        {
                            "model_path": settings.llm_model,
                            "prompt_style": settings.llm_prompt_style,
                        }
                        if settings.use_llm_rewriter
                        else None
                    )
                    self._pipeline = ArchiveDescriptionPipeline(
                        model_path=settings.caption_model,
                        caption_kwargs={
                            "backend": settings.caption_backend,
                            "num_beams": 1,
                            "length_penalty": 1.0,
                            "max_new_tokens": 50,
                        },
                        taxonomy_version=settings.taxonomy_version,
                        use_llm_rewriter=settings.use_llm_rewriter,
                        llm_rewriter_kwargs=llm_kwargs,
                        use_ocr=settings.use_ocr,
                        use_clipscore=settings.use_clipscore,
                    )

    def infer(self, image_path: str) -> dict:
        logging.info("Running inference on %s", image_path)
        self._ensure_pipeline()
        res = self._pipeline.run(image_path)
        logging.info("Returning result %s", res)
        return res

    def warmup(self) -> None:
        """Загрузить пайплайн заранее (вызывается в фоне при старте приложения)."""
        try:
            logging.info("Warmup: loading pipeline...")
            self._ensure_pipeline()
            logging.info("Warmup: pipeline ready.")
        except Exception:
            logging.exception("Warmup failed; модели загрузятся при первом запросе.")


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService()
