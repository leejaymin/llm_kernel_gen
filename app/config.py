from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./tasks.db"
    API_VERSION: str = "v1"
    APP_TITLE: str = "Task Management API"
    APP_VERSION: str = "1.0.0"
    DEFAULT_PAGE_LIMIT: int = 20
    MAX_PAGE_LIMIT: int = 100

    model_config = {"env_prefix": "TASK_API_", "env_file": ".env"}


settings = Settings()
