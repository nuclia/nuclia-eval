from pydantic import BaseModel, Field

CONTEXT_RELEVANCE_TEMPLATE = """\
You are a RELEVANCE grader, tasked with providing relevance scores between a given QUESTION and the CONTEXT provided.
Respond by reporting the context relevance metric with the provided function, where the score value ranges from 0 (no relevance) to 5 (entirely relevant).

Additional Scoring Guidelines:

- Long and short CONTEXTS should be equally considered for relevance assessment.
- Language differences should not influence the score.
- The relevance score should increase as the CONTEXT provides more relevant information to the QUESTION.
- Higher scores indicate relevance to more parts of the QUESTION.
- A score of 1 indicates relevance to some parts, while 2 or 3 suggests relevance to most parts.
- Scores of 4 or 5 should be reserved for CONTEXT that is relevant to the entire QUESTION, with higher scores indicating greater relevance.
- CONTEXT must be helpful for answering the entire QUESTION to receive a score of 5.


QUESTION:
```
\"""
{query}
\"""
```

CONTEXT:
```
\"""
{context}
\"""
```

CONTEXT RELEVANCE SCORE: """


class ContextRelevanceResponse(BaseModel):
    score: int = Field(
        ge=0, le=5, description="The score of the metric, on a scale of 0 to 5"
    )


GROUNDEDNESS_TOOL = {
    "type": "function",
    "function": {
        "name": "context_relevance",
        "description": "Is the retrieved context relevant to the question?",
        "parameters": {
            "type": "object",
            "properties": ContextRelevanceResponse.model_json_schema()["properties"],
            "required": ContextRelevanceResponse.model_json_schema()["required"],
        },
    },
}
