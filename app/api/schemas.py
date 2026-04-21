from pydantic import BaseModel, Field


class CandidateScore(BaseModel):
    label: str
    score: float


class MetadataField(BaseModel):
    label: str
    score: float
    confident: bool
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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
