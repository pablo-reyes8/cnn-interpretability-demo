
def test_predict_url_missing_or_invalid_scheme(client):
    data = {"url": "ftp://example.com/image.png"}
    r = client.post("/predict", data=data)
    assert r.status_code in (400, 422, 500) 


def test_predict_requires_exactly_one_input(client):
    r = client.post("/predict", data={})
    assert r.status_code == 400
    assert "exactamente uno" in r.json().get("detail", "").lower()