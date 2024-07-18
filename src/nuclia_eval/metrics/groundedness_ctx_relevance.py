# from pydantic import BaseModel, Field

# GROUNDEDNESS_CTX_RELEVANCE_TEMPLATE = """\
# You are a RELEVANCE (context_relevance) grader and an INFORMATION OVERLAP (groundedness) classifier.
# You must respond by reporting the context relevance and groundedness (information overlap) metrics with the provided functions, where the score value ranges from 0 to 5.

# Groundedness is defined as the degree of information overlap to which the STATEMENT contains information that is substantially similar or identical to that in the SOURCE. When evaluating overlap, differences in language, phrasing, or structure should not be considered.

# Groundedness scoring guidelines:

# - SCORE 0: No information overlap
# - SCORE 1: Minimal information overlap
# - SCORE 2: Some information overlap
# - SCORE 3: Moderate information overlap
# - SCORE 4: Extensive information overlap
# - SCORE 5: Complete information overlap

# ---

# The context relevance is the relevance of the SOURCE to the QUESTION provided. Respond by reporting the context relevance metric with the provided function, where the score value ranges from 0 (no relevance) to 5 (entirely relevant).

# Context Relevance Scoring Guidelines:

# - Long and short SOURCES should be equally considered for relevance assessment.
# - Language differences should not influence the score.
# - The relevance score should increase as the SOURCE provides more relevant information to the QUESTION.
# - Higher scores indicate relevance to more parts of the QUESTION.
# - A score of 1 indicates relevance to some parts, while 2 or 3 suggests relevance to most parts.
# - Scores of 4 or 5 should be reserved for SOURCE that is relevant to the entire QUESTION, with higher scores indicating greater relevance.
# - SOURCE must be helpful for answering the entire QUESTION to receive a score of 5.

# QUESTION:
# ```
# \"""
# {query}
# \"""

# STATEMENT:
# ```
# \"""
# {answer}
# \"""
# ```

# SOURCE:
# ```
# \"""
# {context}
# \"""
# ```

# METRIC SCORES: """


# class GroundednessCtxRelevanceResponse(BaseModel):
#     context_relevance_score: int = Field(
#         ge=0,
#         le=5,
#         description="The context relevance is the relevance of the SOURCE to the QUESTION provided, on a scale of 0 to 5",
#     )
#     groundedness_score: int = Field(
#         ge=0,
#         le=5,
#         description="Groundedness is defined as the degree of information overlap to which the STATEMENT contains information that is substantially similar or identical to that in the SOURCE, on a scale of 0 to 5",
#     )
# #

# GROUNDEDNESS_CTX_RELEVANCE_TOOL = {
#     "type": "function",
#     "function": {
#         "name": "grounded_ctx_relevance",
#         "description": "",
#         "parameters": {
#             "type": "object",
#             "properties": GroundednessCtxRelevanceResponse.model_json_schema()[
#                 "properties"
#             ],
#             "required": GroundednessCtxRelevanceResponse.model_json_schema()[
#                 "required"
#             ],
#         },
#     },
# }
