"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    environment: str = "development"
    log_level: str = "INFO"

    # Model hyper-parameters
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    max_sequence_length: int = 50
    top_k: int = 10
    max_top_k: int = 50
    model_bundle_path: str = "models/current"
    max_inference_ms: float = 250.0
    max_concurrent_inferences: int = 8
    cache_ttl_seconds: int = 30
    requests_per_minute: int = 120
    api_key: str | None = None


settings = Settings()
