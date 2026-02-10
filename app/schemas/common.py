from typing import Any, List, Optional
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str


class Pagination(BaseModel):
    offset: int
    limit: int
    total: int


class SuccessResponse(BaseModel):
    status: str = "ok"
    data: Any
    error: None = None


class PaginatedResponse(BaseModel):
    status: str = "ok"
    data: List[Any]
    error: None = None
    pagination: Pagination


class ErrorResponse(BaseModel):
    status: str = "error"
    data: None = None
    error: ErrorDetail
