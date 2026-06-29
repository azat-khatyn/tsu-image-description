from __future__ import annotations

from pathlib import Path

from PIL import Image


# Набор наиболее распространённых соотношений сторон.
# Для вертикальных изображений используются обратные значения.
_STANDARD_RATIOS: dict[str, float] = {
    "1:1": 1.0,
    "5:4": 5 / 4,
    "4:3": 4 / 3,
    "3:2": 3 / 2,
    "16:9": 16 / 9,
}


def extract_image_metadata(image_path: str | Path) -> dict:
    """Возвращает технические характеристики входного изображения."""

    path = Path(image_path)

    with Image.open(path) as image:
        width, height = image.size

    ratio = width / height if height else 0.0
    orientation = _get_orientation(width, height)
    aspect_ratio_label = _nearest_aspect_ratio_label(width, height)

    return {
        "filename": path.name,
        "width": width,
        "height": height,
        "orientation": orientation,
        "aspect_ratio": round(ratio, 4),
        "aspect_ratio_label": aspect_ratio_label,
    }


def _get_orientation(width: int, height: int) -> str:
    """Определяет ориентацию с небольшим допуском для почти квадратных файлов."""

    if width <= 0 or height <= 0:
        return "unknown"

    relative_difference = abs(width - height) / max(width, height)

    # Допуск 2%: например, 1000×1015 будет считаться квадратным.
    if relative_difference <= 0.02:
        return "square"

    return "horizontal" if width > height else "vertical"


def _nearest_aspect_ratio_label(width: int, height: int) -> str:
    """Находит ближайшее стандартное соотношение сторон.

    Метка сохраняет направление изображения:
    3:2 для горизонтального, 2:3 для вертикального.
    """

    if width <= 0 or height <= 0:
        return "unknown"

    ratio = width / height

    if ratio < 1:
        # Для вертикального изображения сравнение выполняется
        # по нормализованному отношению большей стороны к меньшей.
        normalized_ratio = height / width
        label = min(
            _STANDARD_RATIOS,
            key=lambda name: abs(_STANDARD_RATIOS[name] - normalized_ratio),
        )
        left, right = label.split(":")
        return f"{right}:{left}"

    return min(
        _STANDARD_RATIOS,
        key=lambda name: abs(_STANDARD_RATIOS[name] - ratio),
    )
