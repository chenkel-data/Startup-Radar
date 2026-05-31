from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    app_name: str = "Startup Radar"
    app_env: str = Field(default="local", alias="APP_ENV")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    git_sha: str = Field(default="unknown", alias="GIT_SHA")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["pretty", "json"] = Field(default="pretty", alias="LOG_FORMAT")
    log_llm_raw_output: bool = Field(default=False, alias="LOG_LLM_RAW_OUTPUT")
    log_llm_preview_chars: int = Field(
        default=1800, alias="LOG_LLM_PREVIEW_CHARS", ge=200, le=20000
    )
    frontend_origin: str = Field(default="http://localhost:5173", alias="FRONTEND_ORIGIN")

    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="startup-radar", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    apply_schema_on_startup: bool = Field(default=True, alias="APPLY_SCHEMA_ON_STARTUP")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_max_retries: int = Field(default=0, alias="OPENAI_MAX_RETRIES", ge=0, le=5)
    llm_max_concurrency: int = Field(default=3, alias="LLM_MAX_CONCURRENCY", ge=1, le=20)
    llm_timeout_seconds: int = Field(default=45, alias="LLM_TIMEOUT_SECONDS", ge=5)
    llm_retry_attempts: int = Field(default=3, alias="LLM_RETRY_ATTEMPTS", ge=1, le=8)
    llm_gleaning_passes: int = Field(default=1, alias="LLM_GLEANING_PASSES", ge=0, le=3)
    embedding_provider: Literal["openai", "sentence-transformers"] = Field(
        default="openai", alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_st_model: str = Field(default="Qwen/Qwen3-Embedding", alias="EMBEDDING_ST_MODEL")
    embedding_similarity_threshold: float = Field(
        default=0.88, alias="EMBEDDING_SIMILARITY_THRESHOLD", ge=0, le=1
    )
    enable_embedding_resolution: bool = Field(default=True, alias="ENABLE_EMBEDDING_RESOLUTION")

    # ----- MLflow (single source of truth for tracing + tracking + eval) -----
    mlflow_enabled: bool = Field(default=True, alias="MLFLOW_ENABLED")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_public_url: str | None = Field(default=None, alias="MLFLOW_PUBLIC_URL")
    mlflow_experiment_name: str = Field(
        default="startup-radar-ingestion", alias="MLFLOW_EXPERIMENT_NAME"
    )
    mlflow_openai_autolog: bool = Field(default=True, alias="MLFLOW_OPENAI_AUTOLOG")
    mlflow_prompt_extraction_uri: str = Field(
        default="prompts:/article_extraction@champion",
        alias="MLFLOW_PROMPT_EXTRACTION_URI",
    )
    mlflow_prompt_gleaning_uri: str = Field(
        default="prompts:/article_extraction_gleaning@champion",
        alias="MLFLOW_PROMPT_GLEANING_URI",
    )
    mlflow_use_prompt_registry: bool = Field(
        default=True,
        alias="MLFLOW_USE_PROMPT_REGISTRY",
    )

    scrape_timeout_seconds: int = Field(default=20, alias="SCRAPE_TIMEOUT_SECONDS", ge=5)
    max_articles_per_ingest: int = Field(default=150, alias="MAX_ARTICLES_PER_INGEST", ge=1)
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
