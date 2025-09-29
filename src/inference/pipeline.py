import time
from typing import Dict, Literal, Tuple

from PIL import Image

from src.inference.io import load_from_bytes, load_from_path, load_from_url
from src.inference.validate import ensure_valid_image, check_size
from src.inference.preprocess import preprocess_to_tensor


def _prepare_from_bytes(b: bytes):
    raw = load_from_bytes(b)
    img = ensure_valid_image(raw)
    check_size(img, min_size=50)
    t0 = time.perf_counter()
    tensor = preprocess_to_tensor(img)
    t1 = time.perf_counter()

    meta = {
        "width": img.width,
        "height": img.height,
        "inference_input_size": int(tensor.shape[-1]),
        "preprocess_ms": round((t1 - t0) * 1000, 2),}
    
    return {"tensor": tensor, "meta": meta}


def prepare_from_path(path: str):
    """
    Lee imagen desde un path local y devuelve:
      {"tensor": torch.Tensor[1,3,224,224], "meta": {...}}
    """
    
    b = load_from_path(path)
    return _prepare_from_bytes(b)


def prepare_from_url(url: str):
    """
    Descarga imagen desde URL y devuelve:
      {"tensor": torch.Tensor[1,3,224,224], "meta": {...}}
    """

    b = load_from_url(url)
    return _prepare_from_bytes(b)


def prepare_from_bytes(b: bytes):
    """
    Usa bytes en memoria y devuelve:
      {"tensor": torch.Tensor[1,3,224,224], "meta": {...}}
    """

    return _prepare_from_bytes(b)