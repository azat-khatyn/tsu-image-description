from dataclasses import dataclass
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Archive Description MVP")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "tmp" / "uploads"))
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
