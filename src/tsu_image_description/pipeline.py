from .models import CaptionGenerator, Translator
from .siglip_metadata_extractor import SigLIPMetadataExtractor
from .theme_inference import ThemeInferencer
from .description_builder import DescriptionBuilder


class ArchiveDescriptionPipeline:
    def __init__(self):
        self.caption_generator = CaptionGenerator()
        self.translator = Translator()
        self.metadata_extractor = SigLIPMetadataExtractor()
        self.theme_inferencer = ThemeInferencer()
        self.description_builder = DescriptionBuilder()

    def run(self, image_path: str) -> dict:
        caption_en = self.caption_generator.generate(image_path)
        caption_ru = self.translator.translate(caption_en)

        metadata = self.metadata_extractor.extract(image_path)
        inference = self.theme_inferencer.infer(metadata)

        base_result = {
            "caption": {
                "en": caption_en,
                "ru": caption_ru,
            },
            "metadata": metadata,
            "inference": inference,
        }

        description_result = self.description_builder.build(base_result)

        return {
            **base_result,
            **description_result
        }
