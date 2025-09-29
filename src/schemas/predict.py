
from __future__ import annotations

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, confloat



class MetaInfo(BaseModel):
    """Metadatos detallados del request y de la inferencia."""
    width: Optional[int] = Field(None, description="Ancho original de la imagen.")
    height: Optional[int] = Field(None, description="Alto original de la imagen.")
    inference_input_size: Optional[int] = Field(
        None, description="Tamaño de entrada del tensor (e.g., 224).")

    preprocess_ms: Optional[confloat(ge=0)] = Field(
        None, description="Tiempo de preprocesamiento en milisegundos.")

    inference_ms: Optional[confloat(ge=0)] = Field(
        None, description="Tiempo de inferencia en milisegundos.")

    model_version: Optional[str] = Field(None, description="Versión declarada del modelo.")
    device: Optional[str] = Field(None, description="Dispositivo de ejecución (cpu/cuda).")

    source: Optional[Literal["file", "url"]] = Field(
        None, description="Origen del payload recibido.")

    received_mime: Optional[str] = Field(
        None, description="Content-Type recibido (si aplica).")


class PredictResponse(BaseModel):
    """Respuesta estándar de /predict."""
    label: str = Field(..., description="Etiqueta predicha ('cat' o 'dog').")
    scores: Dict[str, float] = Field(
        ..., description="Distribución de probabilidad por clase (softmax).")

    # En v2: usar List[str] + Field(min_length=...) para longitud mínima.
    topk: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        description="Ranking de etiquetas desde la más probable; útil si K>2 (opcional).",
        json_schema_extra={"examples": [["cat", "dog"]]},)

    meta: MetaInfo = Field(
        default_factory=MetaInfo,
        description="Metadatos de preprocesamiento e inferencia.",)
    


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Mensaje de error legible.")


class Artifacts(BaseModel):
    """
    Artefactos de interpretabilidad como data URLs (PNG base64).
    Cada clave puede contener una imagen 'data:image/png;base64,...' o un texto de error.
    """
    kernels_panel: Optional[str] = Field(
        None, description="Panel de kernels (PNG base64).")

    feature_maps_panel: Optional[str] = Field(
        None, description="Panel de feature maps (PNG base64).")

    gradcam_panel: Optional[str] = Field(
        None, description="Panel Grad-CAM (PNG base64).")

    integrated_gradients_overlay: Optional[str] = Field(
        None, description="Overlay de Integrated Gradients (PNG base64).")

    occlusion_overlay: Optional[str] = Field(
        None, description="Overlay de Occlusion Sensitivity (PNG base64).")

    kernels_panel_error: Optional[str] = None
    feature_maps_panel_error: Optional[str] = None
    gradcam_panel_error: Optional[str] = None
    integrated_gradients_error: Optional[str] = None
    occlusion_error: Optional[str] = None




class PredictAdvancedResponse(BaseModel):
    """Respuesta de /predict/advanced: predicción + paneles de interpretabilidad."""
    label: str = Field(..., description="Etiqueta predicha ('cat' o 'dog').")
    scores: Dict[str, float] = Field(
        ..., description="Distribución de probabilidad por clase (softmax).")

    meta: MetaInfo = Field(
        default_factory=MetaInfo,
        description="Metadatos de preprocesamiento e inferencia.")

    artifacts: Artifacts = Field(
        default_factory=Artifacts,
        description="Paneles e imágenes de interpretabilidad en PNG base64.")


