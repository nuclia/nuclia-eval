from nuclia_eval.metrics.base import DiscreteScoreResponse, Metric

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

CONTEXT_RELEVANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "context_relevance",
        "description": "Is the retrieved context relevant to the question?",
        "parameters": {
            "type": "object",
            "properties": DiscreteScoreResponse.model_json_schema()["properties"],
            "required": DiscreteScoreResponse.model_json_schema()["required"],
        },
    },
}

ContextRelevance = Metric(
    template=CONTEXT_RELEVANCE_TEMPLATE,
    response_model=DiscreteScoreResponse,
    tool=CONTEXT_RELEVANCE_TOOL,
)
