from .models import CaptionGenerator, Translator
from .siglip_metadata_extractor import SigLIPMetadataExtractor
from .theme_inference import ThemeInferencer
from .description_builder import DescriptionBuilder
from .text_postprocessor import TextPostprocessor
from .english_caption_postprocessor import EnglishCaptionPostprocessor


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
        self.metadata_extractor = SigLIPMetadataExtractor(taxonomy_version=taxonomy_version)
        self.theme_inferencer = ThemeInferencer()
        self.text_postprocessor = TextPostprocessor()
        self.description_builder = DescriptionBuilder(**(builder_kwargs or {}))
        self.en_postprocessor = EnglishCaptionPostprocessor()

        # Опциональный языковой редактор. Когда включён, заменяет архивное
        # описание от DescriptionBuilder (search_text всё равно берётся из builder).
        self.llm_rewriter = None
        if use_llm_rewriter:
            from .llm_rewriter import LLMRewriter
            self.llm_rewriter = LLMRewriter(**(llm_rewriter_kwargs or {}))

    def run(self, image_path: str) -> dict:
        caption_en_raw = self.caption_generator.generate(image_path)
        caption_en = self.en_postprocessor.clean(caption_en_raw)

        caption_ru_raw = self.translator.translate(caption_en)
        caption_ru = self.text_postprocessor.clean_ru_caption(caption_ru_raw)

        metadata = self.metadata_extractor.extract(image_path)
        inference = self.theme_inferencer.infer(metadata)

        base_result = {
            "caption": {
                "en": caption_en,
                "ru": caption_ru,
                "ru_raw": caption_ru_raw,
            },
            "metadata": metadata,
            "inference": inference,
        }

        description_result = self.description_builder.build(base_result)

        # Замена шаблонного архивного описания на сгенерированное LLM.
        # search_text от builder сохраняется (теги закрытой таксономии для поиска).
        if self.llm_rewriter is not None:
            llm_archive = self.llm_rewriter.rewrite(
                caption_en=caption_en,
                metadata=metadata,
                inference=inference,
            )
            description_result["archive_description_template"] = description_result["archive_description"]
            description_result["archive_description"] = llm_archive

        return {
            **base_result,
            **description_result,
        }
