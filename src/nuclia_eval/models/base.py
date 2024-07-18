from abc import ABC, abstractmethod
from typing import Tuple

from nuclia_eval.metrics.base import DiscreteScoreReasonResponse, DiscreteScoreResponse


class RAGEvaluator(ABC):
    @abstractmethod
    def evaluate_rag(
        self, query: str, answer: str, contexts: list[str]
    ) -> Tuple[
        DiscreteScoreReasonResponse,
        list[DiscreteScoreResponse],
        list[DiscreteScoreResponse],
    ]: ...

    @abstractmethod
    def answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> DiscreteScoreReasonResponse: ...

    # @abstractmethod
    # def groundedness_ctx_relevance(
    #     self, query: str, answer: str, contexts: list[str]
    # ) -> list[GroundednessCtxRelevanceResponse]: ...

    @abstractmethod
    def context_relevance(
        self, query: str, contexts: list[str]
    ) -> list[DiscreteScoreResponse]: ...

    @abstractmethod
    def groundedness(
        self, answer: str, contexts: list[str]
    ) -> list[DiscreteScoreResponse]: ...
