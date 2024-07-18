from pydantic import BaseModel, Field

GROUNDEDNESS_TEMPLATE = """\
You are an INFORMATION OVERLAP classifier. Your task is to determine the extent of overlap between the information in the STATEMENT and the SOURCE.

Information overlap is defined as the degree to which the STATEMENT contains information that is substantially similar or identical to that in the SOURCE. When evaluating overlap, differences in language, phrasing, or structure should not be considered.

Respond by reporting the groundedness metric with the provided function

Groudedness scoring guidelines, the score values range from 0 to 5:

- SCORE 0: No information overlap
- SCORE 1: Minimal information overlap
- SCORE 2: Some information overlap
- SCORE 3: Moderate information overlap
- SCORE 4: Extensive information overlap
- SCORE 5: Complete information overlap


STATEMENT: 
```
\"""
{answer}
\"""
```

SOURCE:
```
\"""
{context}
\"""
```

GROUNDEDNESS SCORE: """


class GroundednessResponse(BaseModel):
    score: int = Field(
        ge=0, le=5, description="The score of the metric, on a scale of 0 to 5"
    )


GROUNDEDNESS_TOOL = {
    "type": "function",
    "function": {
        "name": "groundedness",
        "description": "Is the answer grounded in any of the provided contexts?",
        "parameters": {
            "type": "object",
            "properties": GroundednessResponse.model_json_schema()["properties"],
            "required": GroundednessResponse.model_json_schema()["required"],
        },
    },
}
