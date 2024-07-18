from abc import ABC, abstractmethod
from typing import Tuple

from nuclia_eval.metrics import (
    AnswerRelevance,
    AnswerRelevanceResponse,
    ContextRelevance,
    ContextRelevanceResponse,
    Groundedness,
    GroundednessResponse,
)


class RAGEvaluator(ABC):
    @abstractmethod
    def evaluate_rag(
        self, query: str, answer: str, contexts: list[str]
    ) -> Tuple[
        AnswerRelevanceResponse,
        list[ContextRelevanceResponse],
        list[GroundednessResponse],
    ]: ...

    @abstractmethod
    def answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> AnswerRelevanceResponse: ...

    # @abstractmethod
    # def groundedness_ctx_relevance(
    #     self, query: str, answer: str, contexts: list[str]
    # ) -> list[GroundednessCtxRelevanceResponse]: ...

    @abstractmethod
    def context_relevance(
        self, query: str, contexts: list[str]
    ) -> list[ContextRelevanceResponse]: ...

    @abstractmethod
    def groundedness(
        self, answer: str, contexts: list[str]
    ) -> list[GroundednessResponse]: ...
