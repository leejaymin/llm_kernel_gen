from typing import Optional
from sqlalchemy import case, func
from sqlalchemy.orm import Session
from app.exceptions import TaskNotFoundError
from app.models.task import Task, TaskStatus, TaskPriority
from app.schemas.task import TaskCreate, TaskUpdate


def create_task(db: Session, task_in: TaskCreate) -> Task:
    task = Task(
        title=task_in.title,
        description=task_in.description,
        status=task_in.status.value,
        priority=task_in.priority.value,
        due_date=task_in.due_date,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_task(db: Session, task_id: int) -> Task:
    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        raise TaskNotFoundError(task_id)
    return task


def list_tasks(
    db: Session,
    status: Optional[TaskStatus] = None,
    priority: Optional[TaskPriority] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[Task], int]:
    query = db.query(Task)

    if status is not None:
        query = query.filter(Task.status == status.value)
    if priority is not None:
        query = query.filter(Task.priority == priority.value)

    total = query.count()

    # Handle sorting
    if sort_by == "priority":
        priority_order = case(
            (Task.priority == "high", 3),
            (Task.priority == "medium", 2),
            (Task.priority == "low", 1),
            else_=0,
        )
        order_col = priority_order
    else:
        order_col = getattr(Task, sort_by, Task.created_at)

    if sort_order == "asc":
        query = query.order_by(order_col.asc() if hasattr(order_col, 'asc') else order_col)
    else:
        query = query.order_by(order_col.desc() if hasattr(order_col, 'desc') else order_col)

    tasks = query.offset(offset).limit(limit).all()
    return tasks, total


def update_task(db: Session, task_id: int, task_in: TaskUpdate) -> Task:
    task = get_task(db, task_id)
    update_data = task_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field in ("status", "priority") and value is not None:
            value = value.value if hasattr(value, "value") else value
        setattr(task, field, value)
    db.commit()
    db.refresh(task)
    return task


def delete_task(db: Session, task_id: int) -> None:
    task = get_task(db, task_id)
    db.delete(task)
    db.commit()
