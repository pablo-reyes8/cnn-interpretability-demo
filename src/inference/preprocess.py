import numpy as np
import torch
from PIL import Image

from src.utils.config import get_config, load_pet_stats


def _resize_then_center_crop(img, target, scale):
    """Resize manteniendo aspecto y luego CenterCrop a target."""

    short_side = int(round(target * scale))
    w, h = img.size
    if w < h:
        new_w = short_side
        new_h = int(round(h * (short_side / w)))
    else:
        new_h = short_side
        new_w = int(round(w * (short_side / h)))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - target) // 2
    top = (new_h - target) // 2
    right = left + target
    bottom = top + target
    img = img.crop((left, top, right, bottom))
    return img


def preprocess_to_tensor(img, stats_path: str | None = None):

    """
    Aplica: resize -> center crop -> to tensor -> normalize (mean/std).
    Devuelve tensor float32 con shape [1, 3, H, W].
    """

    cfg = get_config()
    stats = load_pet_stats(stats_path)
    mean = np.array(stats["loc"], dtype=np.float32)
    std = np.array(stats["scale"], dtype=np.float32)
    size = int(stats.get("img_size", 224))

    img = _resize_then_center_crop(img, target=size, scale=float(cfg.RESIZE_SCALE))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Se esperaba una imagen RGB HxWx3 tras el preprocesamiento.")

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  

    tensor = torch.from_numpy(arr).float().unsqueeze(0)  
    return tensor
