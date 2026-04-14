"""FastAPI middleware for logging, timing, and error handling."""

from __future__ import annotations

import time
import traceback

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from medqa.log import get_logger

logger = get_logger("medqa.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "%s %s -> %d (%.0fms)",
                method, path, response.status_code, elapsed,
            )
            response.headers["X-Process-Time-Ms"] = f"{elapsed:.0f}"
            return response
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "%s %s -> 500 (%.0fms): %s\n%s",
                method, path, elapsed, exc, traceback.format_exc(),
            )
            raise
