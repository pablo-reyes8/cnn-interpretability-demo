import json
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
from PIL import Image
import sys, os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# --------------------------------------------------------------------------------------
# Raíz del proyecto (asume este archivo en <repo>/src/utils/config.py)
# --------------------------------------------------------------------------------------
def _find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

ROOT_DIR: Path = _find_project_root()

# --------------------------------------------------------------------------------------
# Configuración única (NO duplicar clases ni get_config() más abajo)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    ROOT: Path = ROOT_DIR

    # Paths de datos/modelo (resueltos relativo al repo)
    PET_STATS_PATH: Path = ROOT_DIR / "data" / "pet_stats.json"
    MODEL_META_PATH: Path = ROOT_DIR / "models" / "meta.json"                   
    MODEL_WEIGHTS_PATH: Path = ROOT_DIR / "resnet101" / "model_trained" / "ResNet101.pth"
    MODEL_YAML_PATH: Path = ROOT_DIR / "resnet101" / "oxford_pets_binary_resnet101.yaml"
    RESNET_SRC_DIR: Path = ROOT_DIR / "resnet101" / "src"

    # Parámetros de API / IO 
    ALLOWED_EXTS: Tuple[str, ...] = tuple(
        os.environ.get("ALLOWED_EXTS", ".jpg,.jpeg,.png").lower().split(","))
    ALLOWED_MIMES: Tuple[str, ...] = tuple(
        os.environ.get("ALLOWED_MIMES", "image/jpeg,image/png").lower().split(","))
    MAX_IMAGE_MB: int = int(os.environ.get("MAX_IMAGE_MB", "5"))
    TIMEOUT_CONNECT: float = float(os.environ.get("TIMEOUT_CONNECT", "5.0"))
    TIMEOUT_READ: float = float(os.environ.get("TIMEOUT_READ", "10.0"))

    # Preprocesamiento / device policy 
    RESIZE_SCALE: float = float(os.environ.get("RESIZE_SCALE", "1.14"))
    DEVICE_POLICY: str = os.environ.get("DEVICE_POLICY", "auto")  # 'auto' | 'cpu' | 'cuda'

# Singleton en memoria (una sola instancia compartida)
_CFG: Optional[Config] = None

def get_config() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = Config()
    return _CFG


# --------------------------------------------------------------------------------------
# Utilidades de stats (acepta pet_stats.json o meta.json con normalization.mean/std)
# --------------------------------------------------------------------------------------
def load_pet_stats(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Carga estadísticas de normalización desde:
      - data/pet_stats.json con llaves {loc, scale, img_size?}, o
      - models/meta.json con {"normalization": {"mean","std"}, "input_size": ...}
    Devuelve dict normalizado: {"loc": [...], "scale": [...], "img_size": 224}
    """
    cfg = get_config()
    stats_path = Path(path) if path is not None else cfg.PET_STATS_PATH
    if not stats_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de stats en: {stats_path}")
    raw = json.loads(stats_path.read_text(encoding="utf-8"))

    # Caso A: esquema pet_stats.json
    if "loc" in raw and "scale" in raw:
        if "img_size" not in raw:
            raw["img_size"] = 224
        return {"loc": raw["loc"], "scale": raw["scale"], "img_size": int(raw["img_size"])}

    # Caso B: esquema meta.json con normalization
    norm = raw.get("normalization", {})
    mean, std = norm.get("mean"), norm.get("std")
    if isinstance(mean, list) and isinstance(std, list):
        return {
            "loc": mean,
            "scale": std,
            "img_size": int(raw.get("input_size", 224)),}

    raise ValueError("El archivo de stats no contiene 'loc/scale' ni 'normalization.mean/std'.")


def mostrar_imagen(ruta):
    """
    Muestra la imagen ubicada en `ruta`.
    Acepta formatos comunes (jpg, png, jpeg, etc.).
    """
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No encontré el archivo: {ruta}")
    try:
        img = Image.open(ruta).convert("RGB")
    except OSError as e:
        raise ValueError(f"El archivo no parece ser una imagen válida:\n{e}")

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(os.path.basename(ruta))
    plt.axis("off")
    plt.show()