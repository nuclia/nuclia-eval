from typing import Any, Type

from pydantic import BaseModel, Field


class Metric(BaseModel):
    """Base class for all metrics"""

    template: str = Field(
        description="The template that will be used to create the user prompt, it can contain {query}, {answer} or/and {context}"
    )
    response_model: Type[BaseModel] = Field(
        description="The output we expect from the model"
    )
    tool: dict[str, Any] = Field(
        description="The tool definition that will given to the model to generate the output"
    )


class DiscreteScoreResponse(BaseModel):
    score: int = Field(
        ge=0, le=5, description="The score of the metric, on a scale of 0 to 5"
    )


class DiscreteScoreReasonResponse(DiscreteScoreResponse):
    reason: str = Field(
        description="The reason for the score, limited to 150 characters"
    )
