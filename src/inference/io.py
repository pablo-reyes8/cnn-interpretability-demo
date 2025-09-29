
import os
from typing import Optional
from src.utils.config import get_config

try:
    import requests 
except Exception:  
    requests = None 

from urllib.parse import urlparse
from urllib.request import urlopen, Request 


def _read_file_bytes(path: str):
    with open(path, "rb") as f:
        return f.read()
    

def load_from_path(path: str) -> bytes:
    """
    Lee un archivo local y devuelve los bytes.
    Errores: FileNotFoundError, PermissionError, IsADirectoryError.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    if not os.path.isfile(path):
        raise IsADirectoryError(f"No es un archivo regular: {path}")
    return _read_file_bytes(path)

def load_from_bytes(b: bytes) -> bytes:
    """
    Devuelve los mismos bytes tras validar tipo.
    Error: TypeError si no es bytes.
    """
    if not isinstance(b, (bytes, bytearray)):
        raise TypeError("Se esperaba 'bytes' o 'bytearray'.")
    return bytes(b)


def _ensure_http_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("Solo se permiten URLs http/https.")
    

def _check_content_length(cl, max_mb):
    if cl is None:
        return
    try:
        size_bytes = int(cl)
    except Exception:
        return
    if size_bytes > max_mb * 1024 * 1024:
        raise ValueError(f"Archivo remoto excede {max_mb} MB.")
    

def load_from_url(url: str, timeout: Optional[int] = None, max_mb: Optional[int] = None) -> bytes:
    """
    Descarga bytes desde un URL con límite de tamaño y timeouts.
    Errores: ValueError (esquema inválido, tamaño), TimeoutError, OSError en red.
    """

    cfg = get_config()
    _ensure_http_url(url)

    t_conn = cfg.TIMEOUT_CONNECT
    t_read = cfg.TIMEOUT_READ
    if timeout is not None:
        t_conn = t_read = float(timeout)

    limit_mb = int(max_mb) if max_mb is not None else int(cfg.MAX_IMAGE_MB)
    ua = "Mozilla/5.0 (CatDog-Classifier/1.0; +local)"

    if requests is not None:
        try:
            with requests.get(
                url,
                headers={"User-Agent": ua},
                stream=True,
                timeout=(t_conn, t_read),) as r:

                r.raise_for_status()
                _check_content_length(r.headers.get("Content-Length"), limit_mb)
                buf = bytearray()

                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        buf.extend(chunk)
                        if len(buf) > limit_mb * 1024 * 1024:
                            raise ValueError(f"Descarga excede {limit_mb} MB.")
                return bytes(buf)
        except requests.exceptions.Timeout as e:  # type: ignore
            raise TimeoutError(f"Timeout descargando {url}") from e
        except requests.exceptions.RequestException as e:  # type: ignore
            raise OSError(f"Error de red al descargar {url}: {e}") from e
        
    else: 
        try:  
            req = Request(url, headers={"User-Agent": ua})
            with urlopen(req, timeout=t_read) as resp:
                _check_content_length(resp.headers.get("Content-Length"), limit_mb)
                buf = bytearray()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if len(buf) > limit_mb * 1024 * 1024:
                        raise ValueError(f"Descarga excede {limit_mb} MB.")
                return bytes(buf)
            
        except Exception as e:  
            raise OSError(f"Error de red al descargar {url}: {e}") from e