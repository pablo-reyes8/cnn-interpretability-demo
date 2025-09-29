# 🐾 ResNet-101: Cat vs Dog Classification with Interpretability

![Repo size](https://img.shields.io/github/repo-size/SPMINE-2425/proyecto-final-reyes-castano)
![Last commit](https://img.shields.io/github/last-commit/SPMINE-2425/proyecto-final-reyes-castano)
![Open issues](https://img.shields.io/github/issues/SPMINE-2425/proyecto-final-reyes-castano)
![Contributors](https://img.shields.io/github/contributors/SPMINE-2425/proyecto-final-reyes-castano)
![Forks](https://img.shields.io/github/forks/SPMINE-2425/proyecto-final-reyes-castano?style=social)
![Stars](https://img.shields.io/github/stars/SPMINE-2425/proyecto-final-reyes-castano?style=social)

A full-stack repository (**API + App**) that implements **ResNet-101** in PyTorch to classify images of **cats** and **dogs**, and includes an **interpretability** module to understand what the network “sees” for each prediction.  
The goal is to provide an advanced, reliable classification and explanation tool with **high reproducibility**: modular code, YAML-based configuration, and evaluation/visualization utilities.

---

## 🎯 Objectives

- **Binary classification** (cat vs. dog) on user-supplied images (local file or URL).
- **Advanced interpretability** to inspect the model’s decision per image:
  - **Occlusion Sensitivity** — patch-wise relevance mapping.
  - **Integrated Gradients** — path-integrated attributions from a baseline.
  - **Grad-CAM** — class-specific heatmaps.
  - **Feature Maps (by depth)** — intermediate activations per layer.
  - **Kernels (by depth)** — learned filters (e.g., first conv layer).

---

## 🧩 Main Components

- **`resnet101/`** — model implementation (from scratch) and experiment artifacts

  - `src/` — architecture, residual blocks, checkpoint utilities
  - `model_trained/` — trained weights
  - `experiments/` — results and generated dashboards

- **`src/`** — inference API (FastAPI) and utilities

  - `api/` — routers (`/health`, `/predict`, `/predict/advanced`), errors, middleware, deps
  - `inference/` — preprocessing → forward pass → postprocessing → validation pipeline
  - `schemas/` — Pydantic v2 contracts for requests/responses (incl. metadata & base64 images)
  - `utils/` — configuration, env var loading, path helpers, etc.
  - `tests/` — contract tests and input validation (pytest)

- **`app/`** — user interface (Streamlit) to upload images/URLs and explore explanations

- **`data/`** — data prep and statistics

  - `processed/` — working directory (do not version raw data)
  - `pet_stats.json` — means/standard deviations for reproducible normalization

- **`notebooks/`** — data-flow verification and model sanity checks

- **`oxford_pets_binary_resnet101.yaml`** — experiment configuration (data, model, optimizer, scheduler, device)

---

## 🔐 Weights & Data

- **Model weights** must be downloaded from (...) and placed in the path: **`resnet101/model_trained`**.
- **Datasets** must comply with their original licenses. This project uses **Oxford-IIIT Pet** strictly for educational purposes.

---

## 🖼️ Showcase

**App overview (Home / Image upload):**

<div align="center">
  <img src="app/showcase/Showcase app.png" alt="showcase app" width="580">
</div>

---

**Advanced Prediction (Method-specific Interpretability):**

<div align="center">
  <strong>Grad-CAM</strong><br>
  <img src="advance_visualization/samples/Grad Cam.png" alt="Grad-CAM" width="620"><br>
  <sub>Class-specific heatmap.</sub>
</div>

---

<div align="center">
  <table>
    <tr>
      <td align="center" style="padding:12px;">
        <strong>Occlusion Sensitivity</strong><br>
        <img src="advance_visualization/samples/oclusion v2.png" alt="Occlusion Sensitivity" width="290"><br>
        <sub>Local relevance by hiding patches.</sub>
      </td>
      <td align="center" style="padding:12px;">
        <strong>Integrated Gradients (overlay)</strong><br>
        <img src="advance_visualization/samples/Integrated Gradients.png" alt="Integrated Gradients overlay" width="290"><br>
        <sub>Accumulated attributions overlaid on the input.</sub>
      </td>
    </tr>
  </table>
</div>

---

<div align="center">
  <strong>Feature Maps (depth/layers)</strong><br>
  <img src="advance_visualization/samples/Feature Maps.png" alt="Feature Maps" width="620"><br>
</div>

<div align="center" style="padding-top:8px;">
  <strong>Kernels (learned filters)</strong><br>
  <img src="advance_visualization/samples/Filters.png" alt="Learned Kernels" width="620"><br>
  <sub>Early-layer filters (edges, textures, orientations).</sub>
</div>

---

## 🔧 Installation & Execution

> Minimum requirements: **Python 3.11+** (if using Poetry), **Git**.  
> Recommended alternative: **Docker** + **Docker Compose** (no need to install Python or Poetry locally).

### 1) Clone the repository

```bash
git clone <URL>
cd <PROYECT CARPET>
```

### 2) Model Weights

Place the weights file (e.g., `ResNet101.pth`) in: **`resnet101/model_trained/`**

---

## Option A — Run with Poetry (local)

> Useful for quick development without containers.

1. Install dependencies and activate the virtual environment:

```bash
poetry install
```

2. (Optional) Environment variable to make Streamlit point to a different API:

```bash
# PowerShell
$env:API_BASE_URL="http://127.0.0.1:8000"

# Bash
export API_BASE_URL="http://127.0.0.1:8000"
```

3. Open **two terminals** at the project root:

**Terminal 1 — API (FastAPI/Uvicorn)**

```bash
poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — UI (Streamlit)**

```bash
poetry run streamlit run app/app.py
```

- API docs: http://localhost:8000/docs
- UI: http://localhost:8501

---

## Option B — Run with Docker / Docker Compose (recommended)

> Spin up **API** and **UI** in separate containers with a single command.

### B.1 Build the image (if not already built)

```bash
docker build -t mi_app:latest_final .
```

### B.2 Use `docker-compose.yml`

```bash
docker compose up -d
```

- API: http://localhost:8000/docs
- UI: http://localhost:8501

### B.3 Useful Commands

```bash
# Start in detached mode
docker compose up -d

# View logs (API / UI)
docker compose logs -f api
docker compose logs -f ui

# Rebuild & restart after code changes
docker compose up -d --build

# Stop and remove containers
docker compose down
```

### 🛠️ Common Issues

- **Weights not found**: check the path `resnet101/model_trained/ResNet101.pth`.
- **CORS / App–API connection**: verify `API_BASE_URL` and ensure Uvicorn is running at `127.0.0.1:8000`.
- **Dependencies**: reinstall with `poetry install` or update with `poetry update`.

---

## 🗺️ Exposed Routes (API)

- `GET /health` → Service status
- `POST /predict` → Basic prediction: `label`, `scores`, `meta`
- `POST /predict/advanced` → Prediction + interpretability `artifacts`

---

## 🔍 Inference Flow

1. **Input**: file or URL → validation (MIME/shape)
2. **Preprocessing**: resize/center-crop → normalized tensor (cached statistics)
3. **Model (ResNet-101)**: forward pass → logits → softmax
4. **Basic output**: `label` + `scores` + `meta`
5. **Advanced output**: adds `artifacts` with panels (base64 PNGs) for:
   - `gradcam_panel`
   - `integrated_gradients_overlay`
   - `occlusion_overlay`
   - `feature_maps_panel`
   - `kernels_panel`  
     (includes error indicators per panel when applicable)

---

## 📊 Benchmark Metrics (Validation)

<div align="center">

<p><em>Summary</em></p>

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Val Loss</td>
      <td><strong>0.4084</strong></td>
    </tr>
    <tr>
      <td>ROC-AUC</td>
      <td><strong>0.9108</strong></td>
    </tr>
  </tbody>
</table>

<br/>

<p><em>Classification Report</em></p>

<table>
  <thead>
    <tr>
      <th>Class</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0 (cat)</td>
      <td>0.6840</td>
      <td>0.8750</td>
      <td>0.7678</td>
      <td>240</td>
    </tr>
    <tr>
      <td>1 (dog)</td>
      <td>0.9301</td>
      <td>0.8044</td>
      <td>0.8627</td>
      <td>496</td>
    </tr>
    <tr>
      <td><strong>Accuracy</strong></td>
      <td></td>
      <td></td>
      <td><strong>0.8274</strong></td>
      <td>736</td>
    </tr>
    <tr>
      <td><strong>Macro Avg</strong></td>
      <td>0.8071</td>
      <td>0.8397</td>
      <td>0.8153</td>
      <td>736</td>
    </tr>
    <tr>
      <td><strong>Weighted Avg</strong></td>
      <td>0.8498</td>
      <td>0.8274</td>
      <td>0.8318</td>
      <td>736</td>
    </tr>
  </tbody>
</table>

</div>

---

## ⚙️ Reproducibility

- Centralized **YAML** config (dataset, normalization, architecture, optimizer, scheduler).
- Normalization statistics cached in `data/pet_stats.json`.
- Controlled seeds and devices (CPU/CUDA).

---

## 📄 License & Credits

- Intended for **educational and research** use.
- Please cite **He et al., 2016** (ResNet) and the original interpretability works used in this project.
- Thanks to the PyTorch community and the Oxford-IIIT Pet dataset.

---

## 📬 Contact

- For issues and enhancements, use the repository’s **Issue Tracker**.
- For technical questions, open a **Discussion** with the `help-wanted` tag.
