from datetime import date, timedelta


def _create_task(client, **overrides):
    payload = {"title": "Test Task", **overrides}
    response = client.post("/api/v1/tasks", json=payload)
    return response


class TestCreateTask:
    def test_create_task_minimal(self, client):
        response = _create_task(client)
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "ok"
        assert data["data"]["title"] == "Test Task"
        assert data["data"]["status"] == "pending"
        assert data["data"]["priority"] == "medium"
        assert data["data"]["description"] is None
        assert data["data"]["due_date"] is None
        assert data["data"]["id"] is not None
        assert data["data"]["created_at"] is not None
        assert data["data"]["updated_at"] is not None
        assert data["error"] is None

    def test_create_task_full(self, client):
        future_date = (date.today() + timedelta(days=7)).isoformat()
        response = _create_task(
            client,
            title="Full Task",
            description="A detailed description",
            status="in_progress",
            priority="high",
            due_date=future_date,
        )
        assert response.status_code == 201
        data = response.json()["data"]
        assert data["title"] == "Full Task"
        assert data["description"] == "A detailed description"
        assert data["status"] == "in_progress"
        assert data["priority"] == "high"
        assert data["due_date"] == future_date

    def test_create_task_empty_title(self, client):
        response = client.post("/api/v1/tasks", json={"title": ""})
        assert response.status_code == 422
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_create_task_whitespace_title(self, client):
        response = client.post("/api/v1/tasks", json={"title": "   "})
        assert response.status_code == 422
        data = response.json()
        assert data["status"] == "error"

    def test_create_task_no_title(self, client):
        response = client.post("/api/v1/tasks", json={})
        assert response.status_code == 422

    def test_create_task_past_due_date(self, client):
        past_date = (date.today() - timedelta(days=1)).isoformat()
        response = _create_task(client, due_date=past_date)
        assert response.status_code == 422
        data = response.json()
        assert data["status"] == "error"


class TestGetTask:
    def test_get_task(self, client):
        create_resp = _create_task(client, title="Get Me")
        task_id = create_resp.json()["data"]["id"]

        response = client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["data"]["title"] == "Get Me"
        assert data["data"]["id"] == task_id

    def test_get_task_not_found(self, client):
        response = client.get("/api/v1/tasks/9999")
        assert response.status_code == 404
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["code"] == "TASK_NOT_FOUND"


class TestListTasks:
    def test_list_tasks_empty(self, client):
        response = client.get("/api/v1/tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["data"] == []
        assert data["pagination"]["total"] == 0

    def test_list_tasks_pagination(self, client):
        for i in range(25):
            _create_task(client, title=f"Task {i}")

        response = client.get("/api/v1/tasks?offset=0&limit=10")
        data = response.json()
        assert len(data["data"]) == 10
        assert data["pagination"]["total"] == 25
        assert data["pagination"]["offset"] == 0
        assert data["pagination"]["limit"] == 10

        response2 = client.get("/api/v1/tasks?offset=20&limit=10")
        data2 = response2.json()
        assert len(data2["data"]) == 5
        assert data2["pagination"]["total"] == 25

    def test_list_tasks_filter_status(self, client):
        _create_task(client, title="Pending", status="pending")
        _create_task(client, title="Done", status="completed")

        response = client.get("/api/v1/tasks?status=pending")
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["title"] == "Pending"

    def test_list_tasks_filter_priority(self, client):
        _create_task(client, title="Low", priority="low")
        _create_task(client, title="High", priority="high")

        response = client.get("/api/v1/tasks?priority=high")
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["title"] == "High"

    def test_list_tasks_sort_asc(self, client):
        _create_task(client, title="Alpha")
        _create_task(client, title="Beta")

        response = client.get("/api/v1/tasks?sort_by=title&sort_order=asc")
        data = response.json()
        assert data["data"][0]["title"] == "Alpha"
        assert data["data"][1]["title"] == "Beta"


class TestUpdateTask:
    def test_update_task_partial(self, client):
        create_resp = _create_task(client, title="Original", description="Desc")
        task_id = create_resp.json()["data"]["id"]

        response = client.patch(f"/api/v1/tasks/{task_id}", json={"title": "Updated"})
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["title"] == "Updated"
        assert data["description"] == "Desc"  # unchanged

    def test_update_task_clear_description(self, client):
        create_resp = _create_task(client, title="Task", description="Has desc")
        task_id = create_resp.json()["data"]["id"]

        response = client.patch(f"/api/v1/tasks/{task_id}", json={"description": None})
        assert response.status_code == 200
        assert response.json()["data"]["description"] is None

    def test_update_task_not_found(self, client):
        response = client.patch("/api/v1/tasks/9999", json={"title": "Nope"})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "TASK_NOT_FOUND"


class TestDeleteTask:
    def test_delete_task(self, client):
        create_resp = _create_task(client)
        task_id = create_resp.json()["data"]["id"]

        response = client.delete(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 204

        get_resp = client.get(f"/api/v1/tasks/{task_id}")
        assert get_resp.status_code == 404

    def test_delete_task_not_found(self, client):
        response = client.delete("/api/v1/tasks/9999")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "TASK_NOT_FOUND"
