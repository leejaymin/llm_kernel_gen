from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check():
    return {
        "status": "ok",
        "data": {
            "service": "task-management-api",
            "version": "1.0.0",
        },
        "error": None,
    }
