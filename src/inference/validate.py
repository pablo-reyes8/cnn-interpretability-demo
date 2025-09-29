import io
from PIL import Image, ImageOps 
from src.utils.config import get_config


class InvalidImageError(ValueError):
    pass


def _infer_mime_from_pil_format(fmt: str | None):
    if fmt is None:
        return None
    fmt = fmt.upper()
    if fmt == "JPEG":
        return "image/jpeg"
    if fmt == "PNG":
        return "image/png"
    return None


def ensure_valid_image(data: bytes):
    """
    Valida que los bytes representen una imagen JPEG o PNG,
    corrige orientación EXIF y convierte a RGB.
    """

    cfg = get_config()
    try:
        img = Image.open(io.BytesIO(data))
        img.load() 
    except Exception as e:
        raise InvalidImageError("No se pudo decodificar la imagen.") from e

    mime = _infer_mime_from_pil_format(img.format)
    if mime is None or mime.lower() not in cfg.ALLOWED_MIMES:
        raise InvalidImageError("Formato no soportado. Solo JPEG/PNG.")

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img



def check_size(img, min_size= 50) :
    """
    Verifica que la imagen tenga al menos min_size x min_size píxeles.
    """

    w, h = img.size
    if w < min_size or h < min_size:
        raise InvalidImageError(
            f"Imagen demasiado pequeña ({w}x{h}). Mínimo {min_size}x{min_size}.")
    

