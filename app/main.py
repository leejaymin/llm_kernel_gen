from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from app.config import settings
from app.database import Base, engine
from app.exceptions import (
    TaskNotFoundError,
    task_not_found_handler,
    validation_error_handler,
    generic_error_handler,
)
from app.routes import health_routes, task_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Exception handlers
app.add_exception_handler(TaskNotFoundError, task_not_found_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, generic_error_handler)

# Routes
app.include_router(health_routes.router, prefix=f"/api/{settings.API_VERSION}")
app.include_router(task_routes.router, prefix=f"/api/{settings.API_VERSION}")
