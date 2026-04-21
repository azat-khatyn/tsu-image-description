from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.api.schemas import HealthResponse, InferenceResponse
from app.core.config import settings
from app.services.image_io import (
    cleanup_file,
    save_upload_file,
    validate_local_image_path,
)
from app.services.inference_service import InferenceService, get_inference_service


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    html_path = Path(__file__).resolve().parents[1] / "ui" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
def health(
    service: InferenceService = Depends(get_inference_service),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=service.model_loaded,
        device=service.device,
    )


@app.post("/inference", response_model=InferenceResponse)
async def inference(
    file: UploadFile | None = File(default=None),
    image_path: str | None = Form(default=None),
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    saved_path = None
    final_image_path: Path | None = None
    filename_for_response: str | None = None

    has_file = file is not None and bool(file.filename)
    has_image_path = image_path is not None and bool(image_path.strip())

    if has_file and has_image_path:
        raise HTTPException(
            status_code=400,
            detail="Передай либо file, либо image_path, но не оба сразу.",
        )

    if not has_file and not has_image_path:
        raise HTTPException(
            status_code=400,
            detail="Нужно передать либо file, либо image_path.",
        )

    try:
        if has_file:
            saved_path = await save_upload_file(
                upload_file=file,
                upload_dir=settings.upload_dir,
                max_upload_size_mb=settings.max_upload_size_mb,
                allowed_extensions=settings.allowed_extensions,
            )
            final_image_path = saved_path
            filename_for_response = file.filename or saved_path.name
        else:
            validated_path = validate_local_image_path(
                image_path=image_path.strip(),
                allowed_extensions=settings.allowed_extensions,
            )
            final_image_path = validated_path
            filename_for_response = validated_path.name

        result = service.infer(str(final_image_path))
        return InferenceResponse(
            filename=filename_for_response,
            **result,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка во время инференса: {exc}",
        ) from exc
    finally:
        if saved_path is not None:
            cleanup_file(saved_path)
