from abc import ABC, abstractmethod
from typing import Tuple

from nuclia_eval.metrics.base import DiscreteScoreReasonResponse, DiscreteScoreResponse


class RAGEvaluator(ABC):  # pragma: no cover
    """Base class for all RAG evaluators"""

    @abstractmethod
    def evaluate_rag(
        self, query: str, answer: str, contexts: list[str]
    ) -> Tuple[
        DiscreteScoreReasonResponse,
        list[DiscreteScoreResponse],
        list[DiscreteScoreResponse],
    ]:
        """This method evaluates a whole RAG experience, given the user's query, the model's answer and contexts retrieved at the retrieval phase it computes the answer relevance, context relevance and groundedness.

        * Answer relevance refers to the directness and appropriateness of the response in addressing the specific question asked, providing accurate, complete, and contextually suitable information.
        * The context relevance is the relevance of the **context** to the **question**, on a scale of 0 to 5.
        * Groundedness is defined as the degree of information overlap to which the **answer** contains information that is substantially similar or identical to that in the **context** piece. The scores are between 0 and 5.


        For more information on the RAG evaluation, you can see each metric's definition at nuclia-eval/src/nuclia_eval/metrics/

        Args:
            query (str): The user's query
            answer (str): The model's answer to the user's query
            contexts (list[str]): The contexts retrieved at the retrieval phase that were used to generate the answer

        Returns:
            Tuple[ DiscreteScoreReasonResponse, list[DiscreteScoreResponse], list[DiscreteScoreResponse], ]: A tuple containing the evaluation result for the answer relevance, the context relevance results and groundedness results
        """
        ...

    @abstractmethod
    def answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> DiscreteScoreReasonResponse:
        """This method evaluates the relevance of the model's answer to the user's query.

        Answer relevance refers to the directness and appropriateness of the response in addressing the specific question asked, providing accurate, complete, and contextually suitable information.

        For more information on answer relevance, see the metric's definition at nuclia-eval/src/nuclia_eval/metrics/answer_relevance.py

        Args:
            query (str): The user's query
            answer (str): The model's answer to the user's query

        Returns:
            DiscreteScoreReasonResponse: The evaluation result for the answer relevance
        """
        ...

    @abstractmethod
    def context_relevance(
        self, query: str, contexts: list[str]
    ) -> list[DiscreteScoreResponse]:
        """This method evaluates the relevance of the contexts retrieved at the retrieval phase to the user's query.

        The context relevance is the relevance of the **context** to the **question**, on a scale of 0 to 5.

        For more information on context relevance, see the metric's definition at nuclia-eval/src/nuclia_eval/metrics/context_relevance.py

        Args:
            query (str): The user's query
            contexts (list[str]): The contexts retrieved at the retrieval phase

        Returns:
            list[DiscreteScoreResponse]: The evaluation results for the context relevance
        """
        ...

    @abstractmethod
    def groundedness(
        self, answer: str, contexts: list[str]
    ) -> list[DiscreteScoreResponse]:
        """This method evaluates the groundedness of the model's answer to the contexts retrieved at the retrieval phase.
        Groundedness is defined as the degree of information overlap to which the **answer** contains information that is substantially similar or identical to that in the **context** piece. The scores are between 0 and 5.
        For more information on groundedness, see the metric's definition at nuclia-eval/src/nuclia_eval/metrics/groundedness.py

        Args:
            answer (str): The model's answer to the user's query
            contexts (list[str]): The contexts retrieved at the retrieval phase

        Returns:
            list[DiscreteScoreResponse]: The evaluation results for the groundedness
        """
        ...
