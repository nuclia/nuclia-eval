from nuclia_eval.metrics.base import DiscreteScoreReasonResponse, Metric

ANSWER_RELEVANCE_TEMPLATE = """You are a RELEVANCE grader, tasked with assessing the relevance of a given RESPONSE to a given QUERY and providing a score along with a brief REASON. Relevance refers to the directness and appropriateness of the response in addressing the specific question asked, providing accurate, complete, and contextually suitable information.

Respond by reporting the answer relevance metric with the provided function

Additional scoring guidelines:
- Long and short responses should be scored equally.
- Relevance score should increase as the response provides relevant context to more parts of the query.

- SCORE 0: RESPONSE is relevant to none of the QUERY.
- SCORE 1: RESPONSE is relevant to some parts of the QUERY.
- SCORE 2: RESPONSE is relevant to most parts of the QUERY but contains superfluous information.
- SCORE 3: RESPONSE is relevant to almost all parts of the QUERY or to the entire QUERY but contains superfluous information.
- SCORE 4: RESPONSE is relevant to the entire QUERY.
- SCORE 5: RESPONSE is relevant to the entire QUERY and answers it completely.

The REASON should be brief and clear, explaining why the RESPONSE received the given SCORE. If the SCORE is not a 5, the REASON should contain how the ANSWER could be improved to a 5.


QUERY: {query}

RESPONSE: {answer}

ANSWER RELEVANCE: """

ANSWER_RELEVANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "answer_relevance",
        "description": "The relevance of an answer is its directness and appropriateness in addressing the specific question asked, providing accurate, complete, and contextually suitable information. It ensures clarity and specificity, avoiding extraneous details while fully satisfying the inquiry.",
        "parameters": {
            "type": "object",
            "properties": DiscreteScoreReasonResponse.model_json_schema()["properties"],
            "required": DiscreteScoreReasonResponse.model_json_schema()["required"],
        },
    },
}

AnswerRelevance = Metric(
    template=ANSWER_RELEVANCE_TEMPLATE,
    response_model=DiscreteScoreReasonResponse,
    tool=ANSWER_RELEVANCE_TOOL,
)
