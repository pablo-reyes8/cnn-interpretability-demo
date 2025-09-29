
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
import sys, os
import torch
from contextlib import contextmanager
import types
try:
    import yaml 
except Exception:
    yaml = None

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESNET_SRC = ROOT_DIR / "resnet101" / "src"
if RESNET_SRC.exists() and str(RESNET_SRC) not in sys.path:
    sys.path.insert(0, str(RESNET_SRC))

from src.utils.config import get_config


@dataclass
class _APIState:
    model = None
    class_names = None  
    id_to_label = None  
    model_name = "ResNet101"
    model_version = "1.0.0"
    input_size = 224
    device = None
    ready = False


_STATE = _APIState()


def _ensure_import_path():
    """Asegura que <repo>/resnet101/src esté en sys.path para 'model.resnet'."""
    cfg = get_config()
    p = cfg.RESNET_SRC_DIR
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

def _select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_meta(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    
    return {
        "num_classes": 2,
        "class_names": ["cat", "dog"],
        "id_to_label": {"0": "cat", "1": "dog"},
        "input_size": 224,
        "model_name": "ResNet101",
        "model_version": "1.0.0",}


def _read_yaml(path: Path):
    if yaml is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


@contextmanager
def _prefer_resnet_src():
    """
    Prioriza el 'src' del repo resnet101 mientras importamos su módulo.
    Evita la colisión con el 'src' del proyecto principal.
    """
    cfg = get_config()
    resnet_root = cfg.ROOT / "resnet101"         
    resnet_src  = resnet_root / "src"             

    old_sys_path = list(sys.path)
    old_src_module = sys.modules.get("src", None)

    try:
        sys.path.insert(0, str(resnet_root))    
        sys.path.insert(0, str(resnet_src))       

        if old_src_module is not None:
            shadow = types.ModuleType("src")
            shadow.__path__ = [str(resnet_src)]    
            sys.modules["src"] = shadow
        yield

    finally:
        if old_src_module is not None:
            sys.modules["src"] = old_src_module
        else:
            sys.modules.pop("src", None)
        sys.path[:] = old_sys_path


def _import_model_class():
    """
    Importa la clase del modelo desde el repo resnet101 evitando colisión con el 'src' del proyecto.
    Busca primero 'src.model.resnet', y si no, 'model.resnet'. Devuelve ResNet101 o ResNet.
    """

    with _prefer_resnet_src():
        mod = None
        try:
            mod = importlib.import_module("src.model.resnet")
        except ModuleNotFoundError:
            pass

        if mod is None:
            try:
                mod = importlib.import_module("model.resnet")

            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "No se pudo importar ni 'src.model.resnet' ni 'model.resnet'. "
                    "Verifica que exista <repo>/resnet101/src/model/resnet.py"
                ) from e

    if hasattr(mod, "ResNet101"):
        return getattr(mod, "ResNet101")
    if hasattr(mod, "ResNet"):
        return getattr(mod, "ResNet")

    raise ImportError(
        "No se encontró ninguna clase 'ResNet101' ni 'ResNet' en el módulo resnet.")


def _build_model(meta: dict, ycfg: dict):
    """
    Construye el modelo filtrando kwargs del YAML para que solo pasen
    los permitidos por ResNet.__init__.
    """
    
    ModelClass = _import_model_class()
    num_classes = int(meta.get("num_classes", 2))

    raw = {}
    if isinstance(ycfg.get("model"), dict):
        raw = dict(ycfg["model"])

    allowed = {
        "num_classes",
        "first_block",
        "init",
        "bn_eps",
        "bn_momentum",
        "blocks_per_stage",}

    mapping = {
        "blocks": "blocks_per_stage", 
        "depth": None,
        "name": None,
        "pretrained": None,
        "weights": None,
        "input_size": None,}

    kwargs = {}
    for k, v in raw.items():
        if k in mapping:
            if mapping[k] is not None:
                kwargs[mapping[k]] = v
        elif k in allowed:
            kwargs[k] = v

    kwargs["num_classes"] = num_classes

    if "blocks_per_stage" in kwargs:
        bps = kwargs["blocks_per_stage"]
        if isinstance(bps, (list, tuple)):
            kwargs["blocks_per_stage"] = tuple(int(x) for x in bps)
        else:
            kwargs["blocks_per_stage"] = tuple(int(x) for x in str(bps).split(","))
    else:
        if ModelClass.__name__ == "ResNet":
            kwargs["blocks_per_stage"] = (3, 4, 23, 3)

    kwargs.setdefault("first_block", "conv")

    model = ModelClass(**kwargs)
    return model


def _load_state_dict(model, weights_path):
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: {weights_path}")
    
    state = torch.load(str(weights_path), map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[deps] Aviso: faltan claves en state_dict: {missing}")
    if unexpected:
        print(f"[deps] Aviso: claves inesperadas en state_dict: {unexpected}")


def load_resources():
    """Carga única del modelo y metadatos según la estructura del repo."""

    if _STATE.ready:
        return
    cfg = get_config()

    device = _select_device()
    meta = _read_meta(cfg.MODEL_META_PATH)
    ycfg = _read_yaml(cfg.MODEL_YAML_PATH)

    model = _build_model(meta, ycfg)
    _load_state_dict(model, cfg.MODEL_WEIGHTS_PATH)
    model.to(device).eval()

    # Metadatos
    class_names = meta.get("class_names", ["cat", "dog"])
    id_to_label = {int(k): v for k, v in meta.get("id_to_label", {"0": "cat", "1": "dog"}).items()}
    input_size = int(meta.get("input_size", 224))
    model_name = meta.get("model_name", "ResNet101")
    model_version = meta.get("model_version", "1.0.0")

    # Estado global
    _STATE.model = model
    _STATE.class_names = class_names
    _STATE.id_to_label = id_to_label
    _STATE.input_size = input_size
    _STATE.model_name = model_name
    _STATE.model_version = model_version
    _STATE.device = device
    _STATE.ready = True

    print(f"[deps] Cargado {model_name} v{model_version} en {device}; input={input_size}")



def get_model():
    if not _STATE.ready or _STATE.model is None:
        raise RuntimeError("Modelo no cargado. Llama load_resources() en startup.")
    return _STATE.model


def get_class_names():
    return _STATE.class_names or ["cat", "dog"]


def get_id_to_label():
    return _STATE.id_to_label or {0: "cat", 1: "dog"}


def get_model_version():
    return _STATE.model_version


def get_input_size():
    return _STATE.input_size


def get_device():
    return str(_STATE.device) if _STATE.device is not None else "cpu"