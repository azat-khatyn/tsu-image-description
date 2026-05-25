from pydantic import BaseModel, Field


class CandidateScore(BaseModel):
    label: str
    score: float


class MetadataField(BaseModel):
    # label может быть None, когда соответствующая ось таксономии пуста
    # (например, mood в archival_v2 — каталожные описания не содержат
    # субъективных категорий настроения)
    label: str | None = None
    score: float = 0.0
    confident: bool = False
    alternatives: list[CandidateScore] = Field(default_factory=list)


class CaptionBlock(BaseModel):
    en: str
    ru: str


class MetadataBlock(BaseModel):
    image_type: MetadataField
    style: MetadataField
    theme: MetadataField
    mood: MetadataField
    tags: list[str] = Field(default_factory=list)


class InferenceBlock(BaseModel):
    theme: str | None = None
    mood: str | None = None
    theme_confidence: float | None = None
    mood_confidence: float | None = None


class InferenceResponse(BaseModel):
    filename: str
    caption: CaptionBlock
    metadata: MetadataBlock
    inference: InferenceBlock
    archive_description: str
    search_text: str
    # Русифицированные теги уверенно предсказанных полей классификатора.
    # В отличие от metadata.tags (сырые англоязычные метки SigLIP),
    # эти теги пригодны для отображения в UI / архивной системе без перевода.
    tags_ru: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    caption_model: str
    caption_backend: str
    taxonomy_version: str
    use_llm_rewriter: bool
    llm_prompt_style: str | None = None
    llm_model: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    pipeline_config: PipelineConfig | None = None
