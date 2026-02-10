from typing import Optional
from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.task import TaskStatus, TaskPriority
from app.schemas.task import TaskCreate, TaskUpdate, TaskResponse
from app.services import task_service

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", status_code=201)
def create_task(task_in: TaskCreate, db: Session = Depends(get_db)):
    task = task_service.create_task(db, task_in)
    return {
        "status": "ok",
        "data": TaskResponse.model_validate(task).model_dump(mode="json"),
        "error": None,
    }


@router.get("")
def list_tasks(
    status: Optional[TaskStatus] = None,
    priority: Optional[TaskPriority] = None,
    sort_by: str = Query("created_at", pattern="^(created_at|updated_at|due_date|priority|status|title)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    tasks, total = task_service.list_tasks(db, status, priority, sort_by, sort_order, offset, limit)
    return {
        "status": "ok",
        "data": [TaskResponse.model_validate(t).model_dump(mode="json") for t in tasks],
        "error": None,
        "pagination": {
            "offset": offset,
            "limit": limit,
            "total": total,
        },
    }


@router.get("/{task_id}")
def get_task(task_id: int, db: Session = Depends(get_db)):
    task = task_service.get_task(db, task_id)
    return {
        "status": "ok",
        "data": TaskResponse.model_validate(task).model_dump(mode="json"),
        "error": None,
    }


@router.patch("/{task_id}")
def update_task(task_id: int, task_in: TaskUpdate, db: Session = Depends(get_db)):
    task = task_service.update_task(db, task_id, task_in)
    return {
        "status": "ok",
        "data": TaskResponse.model_validate(task).model_dump(mode="json"),
        "error": None,
    }


@router.delete("/{task_id}", status_code=204)
def delete_task(task_id: int, db: Session = Depends(get_db)):
    task_service.delete_task(db, task_id)
    return Response(status_code=204)
