from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image


def _validate_extension(filename: str, allowed_extensions: tuple[str, ...]) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Поддерживаются только файлы: {', '.join(allowed_extensions)}",
        )
    return suffix


def _validate_image_file(file_path: Path) -> None:
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Файл не является корректным изображением.",
        ) from exc


async def save_upload_file(
    upload_file: UploadFile,
    upload_dir: Path,
    max_upload_size_mb: int,
    allowed_extensions: tuple[str, ...],
) -> Path:
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="У файла нет имени.")

    suffix = _validate_extension(upload_file.filename, allowed_extensions)

    content = await upload_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Файл пустой.")

    max_size_bytes = max_upload_size_mb * 1024 * 1024
    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Файл слишком большой. Максимум: {max_upload_size_mb} MB.",
        )

    file_path = upload_dir / f"{uuid4().hex}{suffix}"
    file_path.write_bytes(content)

    try:
        _validate_image_file(file_path)
    except Exception:
        cleanup_file(file_path)
        raise

    return file_path


def validate_local_image_path(
    image_path: str,
    allowed_extensions: tuple[str, ...],
) -> Path:
    if not image_path or not image_path.strip():
        raise HTTPException(
            status_code=400,
            detail="Путь к изображению пустой.",
        )

    path = Path(image_path).expanduser().resolve()

    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Файл не найден: {path}",
        )

    if not path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Указанный путь не является файлом: {path}",
        )

    _validate_extension(path.name, allowed_extensions)
    _validate_image_file(path)

    return path


def cleanup_file(file_path: Path) -> None:
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass
