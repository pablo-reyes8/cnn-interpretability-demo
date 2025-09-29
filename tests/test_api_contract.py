
def test_health_contract(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert "classes" in body

def test_predict_method_not_allowed(client):
    r = client.get("/predict")  
    assert r.status_code == 405

def test_predict_with_file_ok(client, dummy_image_bytes):
    files = {"file": ("cat.png", dummy_image_bytes, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"label", "scores", "meta"}
    assert body["label"] in {"cat", "dog"}
    assert "cat" in body["scores"] and "dog" in body["scores"]

def test_predict_with_bad_mime(client):
    files = {"file": ("note.txt", b"hello", "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code in (400, 415)  

def test_predict_advanced_gradcam_only(client, dummy_image_bytes):
    files = {"file": ("cat.png", dummy_image_bytes, "image/png")}
    data = {"what": "gradcam"}
    r = client.post("/predict/advanced", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    assert "artifacts" in body
    artifacts = body["artifacts"]
    assert ("gradcam_panel" in artifacts) or ("gradcam_panel_error" in artifacts)
