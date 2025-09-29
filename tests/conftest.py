import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.api.main import create_app

@pytest.fixture(scope="session")
def client():
    app = create_app()
    return TestClient(app)

@pytest.fixture
def dummy_image_bytes():
    """Imagen RGB válida (128x128) en memoria para tests."""
    img = Image.new("RGB", (128, 128), (120, 180, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@pytest.fixture(autouse=True)
def patch_light_model(monkeypatch):
    """
    Evita cargar pesos reales y acelera las pruebas.
    Parchamos get_model(), get_id_to_label() y load_resources() si fuera necesario.
    """
    import torch

    class Dummy(torch.nn.Module):
        def forward(self, x):
            b = x.shape[0]
            logits = torch.tensor([[2.0, 1.0]]).repeat(b, 1)
            return logits

    def _dummy_model():
        return Dummy().eval()

    def _dummy_id2label():
        return {0: "cat", 1: "dog"}

    import src.api.deps as deps
    monkeypatch.setattr(deps, "get_model", _dummy_model, raising=True)
    monkeypatch.setattr(deps, "get_id_to_label", _dummy_id2label, raising=True)

    monkeypatch.setattr(deps, "load_resources", lambda: None, raising=True)