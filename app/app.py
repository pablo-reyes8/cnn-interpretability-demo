import io
import json
import os
from typing import Dict, Any, Optional, List
import requests
import streamlit as st
from PIL import Image


# ----------------------------------
# Configuración de cliente
# ----------------------------------
API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_BASE}/predict"
ADVANCED_URL = f"{API_BASE}/predict/advanced"
REQUEST_TIMEOUT_BASIC = (5, 25)    
REQUEST_TIMEOUT_ADV = (5, 250)      

st.set_page_config(page_title="Cat vs Dog — CNN Interpretability Demo", page_icon="🐾", layout="centered")

# ----------------------------------
# Encabezado
# ----------------------------------
st.title("🐾 Cat vs Dog — Interpretabilidad con CNNs")

st.markdown(
    """
    ### 📸 Clasificación automática
    Sube una imagen (**JPG/PNG**) o pega una **URL** y una **ResNet-101** pre-entrenada 
    determinará si es **Gato 🐱** o **Perro 🐶**.

    ---
    ### 🔍 Más que una predicción
    No solo verás el resultado, también podrás **explorar cómo piensa la red** a través de técnicas de interpretabilidad:

    - 🎨 **Grad-CAM** → dónde “mira” la red en la imagen  
    - 🧩 **Feature Maps** → qué patrones internos detecta  
    - 🌀 **Kernels** → filtros que aprende el modelo  
    - 🌈 **Integrated Gradients** → qué píxeles pesan más en la decisión  
    - ⬛ **Occlusion Sensitivity** → impacto al ocultar zonas de la imagen  
    ---
    """)

# ----------------------------------
# Entrada del usuario
# ----------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Sube una imagen (JPEG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
with col2:
    url_input = st.text_input("...o pega una URL http/https", placeholder="https://...")

if uploaded_file and url_input:
    st.info("Usaré **solo** la imagen subida. Borra el archivo si prefieres usar la URL.")
    url_input = ""

st.divider()

# ----------------------------------
# Controles de acción
# ----------------------------------
if "show_adv" not in st.session_state:
    st.session_state.show_adv = False

left, right = st.columns(2)

with left:
    infer_basic = st.button("🔎 Predecir", width="stretch")

with right:
    toggle_adv = st.button(
        ("🧠 Análisis avanzado" if not st.session_state.show_adv else "🧠 Ocultar análisis avanzado"),
        width="stretch")
    if toggle_adv:
        st.session_state.show_adv = not st.session_state.show_adv

infer_advanced = False
if st.session_state.show_adv:
    with st.container(border=True):
        st.write("**Opciones de análisis avanzado**")
        ADV_CHOICES = [
            "gradcam",
            "feature_maps",
            "kernels",
            "integrated_gradients",
            "occlusion",
        ]
        adv_selected: List[str] = st.multiselect(
            "Técnicas a generar",
            options=ADV_CHOICES,
            default=["gradcam", "integrated_gradients", "occlusion"],
            help="Puedes elegir varias; si no eliges ninguna, la API usará 'all'.",
        )
        st.caption("Nota: el análisis avanzado puede tardar (hasta ~2 minutos).")
        infer_advanced = st.button("🚀 Ejecutar análisis avanzado", type="primary", width="stretch")


# ----------------------------------
# Helpers
# ----------------------------------
def _handle_response(resp: requests.Response) -> Dict[str, Any]:
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        raise RuntimeError(f"Respuesta no-JSON de la API (status={resp.status_code}).")

    if resp.status_code != 200:
        detail = payload.get("detail", f"Error {resp.status_code}")
        raise RuntimeError(detail)
    return payload

def call_api_with_file(file, url: str, endpoint: str, is_advanced: bool) -> Dict[str, Any]:
    timeout = REQUEST_TIMEOUT_ADV if is_advanced else REQUEST_TIMEOUT_BASIC
    files = None
    data = None
    if file:
        files = {"file": (file.name, file.getvalue(), file.type or "image/jpeg")}
    elif url:
        data = {"url": url}
    else:
        raise RuntimeError("Falta archivo o URL.")

    resp = requests.post(endpoint, files=files, data=data, timeout=timeout)
    return _handle_response(resp)

def call_api_advanced(file, url: str, what: List[str]) -> Dict[str, Any]:
    timeout = REQUEST_TIMEOUT_ADV
    files = None
    data = {}

    if file:
        files = {"file": (file.name, file.getvalue(), file.type or "image/jpeg")}
    elif url:
        data["url"] = url
    else:
        raise RuntimeError("Falta archivo o URL.")

    data["what"] = "all" if len(what) == 0 else ",".join(what)

    resp = requests.post(ADVANCED_URL, files=files, data=data, timeout=timeout)
    return _handle_response(resp)

def _load_pil_from_upload(file) -> Optional[Image.Image]:
    try:
        return Image.open(io.BytesIO(file.getvalue())).convert("RGB")
    except Exception:
        return None

def _load_pil_from_url(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT_BASIC)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def _ensure_one_input():
    if not uploaded_file and not url_input:
        st.warning("Por favor sube **una** imagen **o** ingresa **una** URL.")
        st.stop()
    if url_input and not (url_input.startswith("http://") or url_input.startswith("https://")):
        st.error("La URL debe comenzar con http:// o https://")
        st.stop()

# ----------------------------------
# Acción: Predicción básica
# ----------------------------------
if infer_basic:
    _ensure_one_input()
    with st.spinner("Procesando predicción..."):
        try:
            if uploaded_file:
                result = call_api_with_file(uploaded_file, "", PREDICT_URL, is_advanced=False)
                preview = _load_pil_from_upload(uploaded_file)
            else:
                result = call_api_with_file(None, url_input, PREDICT_URL, is_advanced=False)
                preview = _load_pil_from_url(url_input)
        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout al comunicar con la API.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("❌ No pude conectar con la API. ¿Ejecutaste `uvicorn src.api.main:app --reload`?")
            st.stop()
        except RuntimeError as e:
            st.error(f"⚠️ Error de la API: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error inesperado: {e}")
            st.stop()

    st.subheader("Resultado")
    label = result.get("label", "?")
    scores = result.get("scores", {})
    meta = result.get("meta", {})

    st.success(f"**Predicción:** {label}")
    if preview is not None:
        st.image(preview, caption="Entrada", use_container_width=True)

    with st.expander("Scores (softmax)"):
        st.json(scores)
    with st.expander("Metadatos"):
        st.json(meta)

# ----------------------------------
# Acción: Predicción avanzada
# ----------------------------------
if infer_advanced:
    _ensure_one_input()
    with st.spinner("Ejecutando análisis avanzado..."):
        try:
            result_adv = call_api_advanced(
                uploaded_file if uploaded_file else None,
                "" if uploaded_file else url_input,adv_selected)
            
            preview = _load_pil_from_upload(uploaded_file) if uploaded_file else _load_pil_from_url(url_input)
        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout (análisis avanzado puede tardar hasta ~2 min).")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("❌ No pude conectar con la API. ¿Ejecutaste `uvicorn src.api.main:app --reload`?")
            st.stop()
        except RuntimeError as e:
            st.error(f"⚠️ Error de la API: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error inesperado: {e}")
            st.stop()

    st.subheader("Resultado (Avanzado)")
    label = result_adv.get("label", "?")
    scores = result_adv.get("scores", {})
    meta = result_adv.get("meta", {})
    artifacts = result_adv.get("artifacts", {})

    st.success(f"**Predicción:** {label}")
    if preview is not None:
        st.image(preview, caption="Entrada", use_container_width=True)

    with st.expander("Scores (softmax)"):
        st.json(scores)
    with st.expander("Metadatos"):
        st.json(meta)

    st.markdown("---")
    st.header("Interpretabilidad de la CNN")

    EXPLAIN = {
        "kernels": (
            "Los **kernels** son filtros que aprende la red para detectar patrones simples como bordes o líneas. "
            "Verlos ayuda a entender **qué patrones busca** el modelo desde el principio."),

        "feature_maps": (
            "Los **feature maps** son las respuestas internas de la red cuando pasa tu imagen por una capa. "
            "Muestran **qué zonas/patrones** activan cada filtro a distintas profundidades (de bordes a partes)."),

        "gradcam": (
            "**Grad-CAM** indica **dónde mira** la red para decidir la clase. Las zonas más calientes contribuyen más a la predicción."),

        "integrated_gradients": (
            "**Integrated Gradients** reparte la 'responsabilidad' de la predicción **píxel por píxel** comparando tu imagen "
            "con una referencia simple y acumulando gradientes; así sabemos **qué píxeles importaron más**."),

        "occlusion": (
            "**Occlusion Sensitivity** tapa **pequeños parches** y mide cuánto cae la probabilidad. "
            "Si al tapar una zona baja mucho, esa zona era **relevante**."),}

    EXPLAIN_KEY_FOR = {
        "kernels_panel": "kernels",
        "feature_maps_panel": "feature_maps",
        "gradcam_panel": "gradcam",
        "integrated_gradients_overlay": "integrated_gradients",
        "occlusion_overlay": "occlusion",}
    

    def show_grid_section(title: str, artifact_key: str, caption: str):
        img = artifacts.get(artifact_key)
        if not img:
            return
        explain_key = EXPLAIN_KEY_FOR[artifact_key]
        st.markdown(f"### {title}")
        st.markdown(EXPLAIN[explain_key])

        _, col_img, _ = st.columns([0.5, 2.5, 0.5])  
        with col_img:
            st.image(img, caption=caption, width="stretch")


    def show_single_overlay(title: str, artifact_key: str, caption: str):
        img = artifacts.get(artifact_key)
        if not img:
            return
        explain_key = EXPLAIN_KEY_FOR[artifact_key]
        st.markdown(f"### {title}")
        st.markdown(EXPLAIN[explain_key])
        _, col_img, _ = st.columns([1, 2, 1])
        with col_img:
            st.image(img, caption=caption, width="stretch")

    show_grid_section("Kernels", "kernels_panel", "Kernels — panel")
    show_grid_section("Feature Maps", "feature_maps_panel", "Feature Maps — panel")
    show_grid_section("Grad-CAM", "gradcam_panel", "Grad-CAM — grid")

    show_single_overlay("Integrated Gradients", "integrated_gradients_overlay", "Integrated Gradients — overlay")
    show_single_overlay("Occlusion Sensitivity", "occlusion_overlay", "Occlusion Sensitivity — overlay")

    err_keys = [k for k in artifacts.keys() if k.endswith("_error")]
    if err_keys:
        with st.expander("Detalles de errores en artefactos"):
            st.json({k: artifacts[k] for k in err_keys})

st.caption(
    "API base: "
    f"`{API_BASE}` · Cambia con la variable de entorno `API_BASE_URL`.")




