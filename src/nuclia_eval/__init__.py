"""nuclia-eval is a library that simplifies evaluating the RAG experience using nuclia's models."""

import logging

logger = logging.getLogger(__name__)

from nuclia_eval.models.remi import REMiEvaluator as REMi  # noqa: E402

__all__ = ["REMi"]
