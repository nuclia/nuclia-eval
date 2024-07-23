from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """General Settings for nuclia-eval, any of these settings can be overridden by providing a custom Settings object to the evaluator or by setting the environment variable with the same name."""

    nuclia_model_cache: str = Field(
        default=str(Path.home().joinpath(".nuclia-model-cache")),
        description="The path to the model cache directory.",
    )
