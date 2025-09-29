
import time
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.deps import load_resources
from src.api.routers.health import router as health_router
from src.api.routers.predict import router as predict_router
from src.api.errors import register_exception_handlers

tags_metadata = [
    {
        "name": "health",
        "description": "Estado del servicio y metadatos del modelo.",
    },
    {
        "name": "predict",
        "description": (
            "**Inferencia básica.** Sube un archivo o pasa una URL y obtén "
            "`label` y `scores` (softmax)."
        ),
    },
    {
        "name": "predict-advanced",
        "description": (
            "**Inferencia avanzada.** Predicción + artefactos de interpretabilidad "
            "(kernels, feature maps, Grad-CAM, Integrated Gradients, Occlusion)."
        ),
    },
]

APP_DESCRIPTION = """
### Cat vs Dog — ResNet101 Inference & Interpretability API

Servicio de **inferencia** y **entendimiento** de una **ResNet-101** entrenada sobre *Oxford-IIIT Pets* (tarea binaria).  
Incluye endpoints para predicción básica y para **interpretabilidad** (Grad-CAM, Integrated Gradients, Occlusion, feature maps y kernels).

**Preprocesamiento** · `resize(1.14×) → center-crop(224) → ToTensor → Normalize(mean/std)`  
**Entradas** · imagen RGB (JPEG/PNG) o URL pública  
**Salidas** · `label ∈ {cat, dog}`, `scores` (softmax), `meta` (timing, device, config)  
**Interpretabilidad** · paneles listos para UI (PNG base64) y medidas auxiliares

> Privacidad: las imágenes se procesan en memoria y **no** se almacenan.
"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="Cat vs Dog Classifier API",
        version="1.0.0",
        description=APP_DESCRIPTION,
        openapi_tags=tags_metadata,
        contact={
            "name": "Equipo MINE IX",
            "email": "alejogranados229@gmail.com",
            "url": "https://www.linkedin.com/in/pablo-alejandro-reyes-granados/",
        },
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        docs_url="/docs",
        redoc_url="/redoc")
    
    register_exception_handlers(app)

    allowed_origins = [
        "http://localhost:8501", "http://127.0.0.1:8501",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost", "http://127.0.0.1",]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],)

    @app.middleware("http")
    async def add_timing_header(request: Request, call_next: Callable):
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0)*1000:.2f}"
        return resp

    @app.on_event("startup")
    def _startup() -> None:
        load_resources()

    app.include_router(health_router)
    app.include_router(predict_router)

    @app.get("/", tags=["health"], summary="Bienvenida")
    def root():
        return {"name": "Cat vs Dog Classifier API",
            "endpoints": ["/health", "/predict", "/predict/advanced", "/docs"],}

    return app

app = create_app()