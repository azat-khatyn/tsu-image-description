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


class OCRBlock(BaseModel):
    # Распознанная надпись (text) и гейт уверенности (confident).
    # В подсказку LLM попадает только text при confident=True.
    text: str = ""
    confidence: float | None = None
    confident: bool = False


class FormatBlock(BaseModel):
    # Ориентация и формат открытки из размеров изображения (без ML).
    orientation: str | None = None   # вертикальная / горизонтальная / квадратная
    aspect: str | None = None        # ближайшее стандартное соотношение, напр. "3:4"
    width: int | None = None
    height: int | None = None


class ReliabilityBlock(BaseModel):
    # Сводка надёжности ответа: level — грубый уровень по доле уверенных осей
    # классификатора; clipscore — reference-free семантическое соответствие
    # описания изображению (None, если стадия выключена).
    level: str = "unknown"
    confident_fields: list[str] = Field(default_factory=list)
    n_confident: int = 0
    n_applicable: int = 0
    mean_confidence: float | None = None
    ocr_confidence: float | None = None
    clipscore: float | None = None


class InferenceResponse(BaseModel):
    filename: str
    caption: CaptionBlock
    metadata: MetadataBlock
    inference: InferenceBlock
    ocr: OCRBlock = Field(default_factory=OCRBlock)
    format: FormatBlock = Field(default_factory=FormatBlock)
    reliability: ReliabilityBlock = Field(default_factory=ReliabilityBlock)
    archive_description: str
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
    use_ocr: bool = False
    use_clipscore: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    # OCR реально инициализирован? None — пайплайн ещё не загружен; False —
    # запрошен, но не поднялся (например, paddle под эмуляцией / нет нативного amd64).
    ocr_available: bool | None = None
    device: str
    pipeline_config: PipelineConfig | None = None


class SaveDescriptionRequest(BaseModel):
    # Ручная правка сотрудником: итоговое описание, надпись и формат по имени файла.
    filename: str
    description: str
    ocr_text: str | None = None
    format: FormatBlock | None = None


class SaveDescriptionResponse(BaseModel):
    saved: bool
    path: str
    saved_at: str
