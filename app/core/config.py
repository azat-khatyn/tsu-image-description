from dataclasses import dataclass
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Archive Description MVP")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "tmp" / "uploads"))
    # Файл с итоговыми описаниями (ручная правка сотрудником). Один JSON-объект,
    # ключ — имя файла изображения; сохранение перезаписывает запись по ключу.
    descriptions_path: Path = Path(
        os.getenv("DESCRIPTIONS_PATH", str(BASE_DIR / "data" / "descriptions" / "descriptions.json"))
    )
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # Конфигурация пайплайна. Значения по умолчанию соответствуют
    # «предложенному решению» из README. Все настройки можно переопределить
    # переменными окружения — это используется в Docker, где LLM-редактор
    # отключается из-за CPU-инференса (см. README → «Контейнеризация»).
    caption_model: str = os.getenv(
        "CAPTION_MODEL", "Salesforce/blip-image-captioning-large"
    )
    caption_backend: str = os.getenv("CAPTION_BACKEND", "blip1")
    taxonomy_version: str = os.getenv("TAXONOMY_VERSION", "archival_v2")
    use_llm_rewriter: bool = _env_bool("USE_LLM_REWRITER", True)
    llm_prompt_style: str = os.getenv("LLM_PROMPT_STYLE", "v1_archival")
    llm_model: str = os.getenv(
        "LLM_MODEL", "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"
    )
    # OCR включён по умолчанию. Backend:
    # auto — PaddleOCR с fallback на Tesseract при ошибке инициализации;
    # paddle — только PaddleOCR;
    # tesseract — только Tesseract.
    use_ocr: bool = _env_bool("USE_OCR", True)
    ocr_backend: str = os.getenv("OCR_BACKEND", "auto").strip().lower()
    # CLIPScore — опциональный reference-free сигнал надёжности (грузит CLIP+M-CLIP).
    use_clipscore: bool = _env_bool("USE_CLIPSCORE", False)


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
