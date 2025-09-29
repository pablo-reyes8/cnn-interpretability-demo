
import logging
import uuid
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.schemas.predict import ErrorResponse  

log = logging.getLogger(__name__)


def _error_payload(detail: str, status_code: int, request: Request) -> Dict[str, Any]:
    """
    Construye el payload de error estandarizado con correlation_id.
    NO expone stack traces al cliente.
    """
    corr_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    return {
        "detail": detail,
        "status_code": status_code,
        "correlation_id": corr_id,
        "path": request.url.path,
        "method": request.method,
    }

def _log_exception(exc: Exception, status_code: int, request: Request) -> None:
    """
    Log estructurado del error (en servidor).
    """
    corr_id = request.headers.get("X-Correlation-ID", "N/A")
    log.error(
        "API error | %s %s | %s | %s | %s",
        request.method,
        request.url.path,
        status_code,
        corr_id,
        repr(exc),
        exc_info=False,)


def register_exception_handlers(app: FastAPI):
    """
    Conecta handlers globales para devolver ErrorResponse consistente.
    Llamar desde main.py después de crear 'app'.
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        status_code = exc.status_code
        detail = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        _log_exception(exc, status_code, request)
        return JSONResponse(
            status_code=status_code,
            content=_error_payload(detail, status_code, request),)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        status_code = 422
        detail = "Error de validación del request."
        _log_exception(exc, status_code, request)
        return JSONResponse(
            status_code=status_code,
            content=_error_payload(detail, status_code, request),)

    @app.exception_handler(TimeoutError)
    async def timeout_exception_handler(request: Request, exc: TimeoutError):
        status_code = 504
        detail = "Timeout procesando la solicitud."
        _log_exception(exc, status_code, request)
        return JSONResponse(
            status_code=status_code,
            content=_error_payload(detail, status_code, request),)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        status_code = 500
        detail = "Error interno del servidor."
        _log_exception(exc, status_code, request)
        return JSONResponse(
            status_code=status_code,
            content=_error_payload(detail, status_code, request),)