from functools import lru_cache
from threading import Lock
import logging

from app.bootstrap import SRC_DIR  # noqa: F401
from tsu_image_description.models import get_device
from tsu_image_description.pipeline import ArchiveDescriptionPipeline


class InferenceService:
    def __init__(self) -> None:
        self._pipeline: ArchiveDescriptionPipeline | None = None
        self._lock = Lock()
        self.device = get_device()

    @property
    def model_loaded(self) -> bool:
        return self._pipeline is not None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    # Safe default config.
                    # If BLIP-2 is too heavy on your M1, switch backend to "blip1"
                    # and optionally point model_path to your best BLIP-1 checkpoint.
                    self._pipeline = ArchiveDescriptionPipeline(
                        model_path="Salesforce/blip-image-captioning-large",
                        caption_kwargs={
                            "backend": "blip1",
                            "num_beams": 1,
                            "length_penalty": 1.0,
                            "max_new_tokens": 50,
                        },
                        builder_kwargs={
                            "template_mode": "full",
                            "include_theme": False,
                            "include_mood": False,
                        },
                    )

    def infer(self, image_path: str) -> dict:
        logging.info("Running inference on %s", image_path)
        self._ensure_pipeline()
        res = self._pipeline.run(image_path)
        logging.info("Returning result %s", res)
        return res


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService()
