def test_health_returns_ok(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["data"]["service"] == "task-management-api"
    assert data["data"]["version"] == "1.0.0"
    assert data["error"] is None
