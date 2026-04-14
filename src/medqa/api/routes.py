"""API route definitions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from medqa import __version__
from medqa.models.schemas import HealthResponse, PipelineRequest, PipelineResponse
from medqa.pipeline.orchestrator import MedQAPipeline

router = APIRouter()

_pipeline: MedQAPipeline | None = None


def get_pipeline() -> MedQAPipeline:
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline


def set_pipeline(pipeline: MedQAPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


@router.get("/health", response_model=HealthResponse)
async def health():
    pipeline = get_pipeline()
    return HealthResponse(
        status="healthy" if pipeline.is_ready else "initializing",
        version=__version__,
        index_size=pipeline.retriever.index_size,
    )


@router.post("/analyze", response_model=PipelineResponse)
async def analyze(request: PipelineRequest):
    pipeline = get_pipeline()
    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Pipeline is still initializing")
    return pipeline.run(request)
