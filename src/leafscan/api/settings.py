"""
src/leafscan/api/settings.py
Environment-based configuration management for LeafScan V3.
Compatible with pydantic-settings when available, with automatic fallback to pydantic.BaseModel.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    from pydantic import BaseModel as BaseSettings
    SettingsConfigDict = None
    HAS_PYDANTIC_SETTINGS = False


class Settings(BaseSettings):
    """LeafScan V3 Configuration Schema."""
    app_env: str = Field(default="production", description="Environment: development, staging, production")
    app_title: str = Field(default="LeafScan V3 — Plant Disease Detection API")
    app_version: str = Field(default="3.0.0")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # CORS
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://localhost:3000"
        ]
    )

    # Resource bounds & intake guards
    max_upload_size_bytes: int = Field(default=10 * 1024 * 1024, description="Max upload size in bytes (10MB)")
    max_decoded_pixels: int = Field(default=40_000_000, description="Max decoded pixels (decompression bomb guard)")
    enable_quality_checks: bool = Field(default=False, description="Enable blur/exposure quality screening")
    blur_threshold: float = Field(default=30.0, description="Laplacian variance threshold for blur rejection")

    # Model & Engine
    model_bundle_path: str = Field(default="models/bundles/v3")
    verify_bundle_checksums: bool = Field(default=True)
    execution_provider: str = Field(default="CPUExecutionProvider")
    intra_op_threads: int = Field(default=2)
    inter_op_threads: int = Field(default=1)

    # Concurrency & Rate Limiting
    max_concurrent_inferences: int = Field(default=4, description="Max concurrent inference calls before 503 load shedding")
    rate_limit_per_minute: int = Field(default=60, description="Allowed requests per minute per IP")
    rate_limiting_enabled: bool = Field(default=True)

    # Integrator API Keys (hashed verification)
    api_key_auth_enabled: bool = Field(default=False)
    authorized_api_key_hashes: List[str] = Field(default_factory=list)

    # Logging & Telemetry
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True)

    if HAS_PYDANTIC_SETTINGS and SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_prefix="LEAFSCAN_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )
    else:
        def __init__(self, **data):
            # Load environment overrides manually when pydantic_settings is not installed
            prefix = "LEAFSCAN_"
            env_data = {}
            for field_name in self.__annotations__:
                env_key = prefix + field_name.upper()
                if env_key in os.environ:
                    val = os.environ[env_key]
                    target_type = self.__annotations__.get(field_name)
                    if target_type == int:
                        env_data[field_name] = int(val)
                    elif target_type == float:
                        env_data[field_name] = float(val)
                    elif target_type == bool:
                        env_data[field_name] = val.lower() in ("true", "1", "yes")
                    elif target_type == List[str]:
                        env_data[field_name] = [item.strip() for item in val.split(",")]
                    else:
                        env_data[field_name] = val
            env_data.update(data)
            super().__init__(**env_data)


_SETTINGS_INSTANCE: Optional[Settings] = None

def get_settings() -> Settings:
    """Returns singleton settings instance."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None:
        _SETTINGS_INSTANCE = Settings()
    return _SETTINGS_INSTANCE
