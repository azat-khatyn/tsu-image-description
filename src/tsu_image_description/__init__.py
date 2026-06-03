"""tsu_image_description — модульный пайплайн архивных описаний открыток.

Публичные модули перечислены в __all__ (без авто-импорта, чтобы `import
tsu_image_description` не тянул тяжёлые зависимости torch/transformers).
"""

__all__ = [
    "pipeline",
    "models",
    "siglip_metadata_extractor",
    "description_builder",
    "caption_cleaner",
    "ocr_extractor",
    "taxonomy",
    "llm_rewriter",
]
