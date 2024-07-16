from abc import ABC, abstractmethod
from typing import Tuple

from nuclia_eval.templates.answer_relevance import AnswerRelevanceResponse
from nuclia_eval.templates.groundedness_ctx_relevance import (
    GroundednessCtxRelevanceResponse,
)


class Evaluator(ABC):
    @abstractmethod
    def evaluate_rag(
        self, query: str, answer: str, contexts: list[str]
    ) -> Tuple[AnswerRelevanceResponse, list[GroundednessCtxRelevanceResponse]]: ...

    @abstractmethod
    def answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> AnswerRelevanceResponse: ...

    @abstractmethod
    def groundedness_ctx_relevance(
        self, query: str, answer: str, contexts: list[str]
    ) -> list[GroundednessCtxRelevanceResponse]: ...
