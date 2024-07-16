from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    nuclia_model_cache: str = Field(
        default=str(Path.home().joinpath(".nuclia-eval-model-cache")),
        description="The path to the model cache directory.",
    )
