"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medqa import __version__
from medqa.api.middleware import RequestLoggingMiddleware
from medqa.api.routes import router, set_pipeline
from medqa.config import get_settings
from medqa.log import get_logger
from medqa.pipeline.orchestrator import MedQAPipeline

logger = get_logger("medqa.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting MedQA API v%s", __version__)

    pipeline = MedQAPipeline(settings)
    pipeline.initialize()
    set_pipeline(pipeline)

    logger.info("Pipeline ready")
    yield
    logger.info("Shutting down MedQA API")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="MedQA API",
        description="Production Medical Question Answering Pipeline",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
