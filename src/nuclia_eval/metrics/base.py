from typing import Any, Type

from pydantic import BaseModel, Field


class Metric(BaseModel):
    template: str
    response_model: Type[BaseModel]
    tool: dict[str, Any]


class DiscreteScoreResponse(BaseModel):
    score: int = Field(
        ge=0, le=5, description="The score of the metric, on a scale of 0 to 5"
    )


class DiscreteScoreReasonResponse(DiscreteScoreResponse):
    reason: str = Field(
        description="The reason for the score, limited to 150 characters"
    )
