import json
from pathlib import Path
from typing import Optional, Tuple, Type, TypeVar

import torch
from huggingface_hub import snapshot_download
from mistral_common.protocol.instruct.messages import (
    ChatMessageType,
    SystemMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, Tool, ToolChoice
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.generate import generate
from mistral_inference.model import Transformer
from pydantic import BaseModel, ValidationError

from nuclia_eval import logger
from nuclia_eval.exceptions import InvalidToolCallException, NoOutputException
from nuclia_eval.models.base import Evaluator
from nuclia_eval.settings import Settings
from nuclia_eval.templates.answer_relevance import (
    ANSWER_RELEVANCE_TEMPLATE,
    ANSWER_RELEVANCE_TOOL,
    AnswerRelevanceResponse,
)
from nuclia_eval.templates.groundedness_ctx_relevance import (
    GROUNDEDNESS_CTX_RELEVANCE_TEMPLATE,
    GROUNDEDNESS_CTX_RELEVANCE_TOOL,
    GroundednessCtxRelevanceResponse,
)
from nuclia_eval.utils import load_lora_low_mem

T = TypeVar("T", bound=BaseModel)


class REMiEvaluator(Evaluator):
    def __init__(
        self,
        settings: Optional[Settings] = None,
        force_download: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        # Load default settings if not provided
        if settings is None:
            self.settings = Settings()

        # Download models
        self._download_base_model(force_download)
        self._download_adapter_model(force_download)

        # Load tokenizer
        logger.info("Loading tokenizer")
        self.tokenizer = MistralTokenizer.from_file(
            str(self._base_model_path / "tokenizer.model.v3")
        )
        logger.info("Tokenizer loaded successfully")

        # Load model on CPU always to avoid memory issues during adapter loading
        logger.info("Loading base model")
        self.model = Transformer.from_folder(
            str(self._base_model_path),
            dtype=torch.float16,  # device="cpu"
        )
        torch.cuda.empty_cache()
        logger.info("Base model loaded successfully")

        # Load LoRA
        logger.info("Loading REMi adapter model")
        load_lora_low_mem(self.model, self._adapter_model_path / "lora.safetensors")
        logger.info("REMi adapter model loaded successfully")

        # Move model to desired device
        self.device = device
        logger.info(f"Moving model to {self.device}")
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to {self.device}")

        # Cache variables
        self._tools: Optional[list[Tool]] = None
        self._system_message: Optional[SystemMessage] = None

    def _download_base_model(self, force_download: bool = False):
        self._base_model_path = (
            Path(self.settings.nuclia_model_cache) / "Mistral-7B-Instruct-v0.3"
        )
        if force_download or not self._base_model_path.exists():
            logger.info(
                f"Downloading base model to {self._base_model_path}, to override this behavior, please provide the path in the settings or as an environment variable `NUCLIA_EVAL_MODEL_CACHE`"
            )
            snapshot_download(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                allow_patterns=[
                    "params.json",
                    "consolidated.safetensors",
                    "tokenizer.model.v3",
                ],
                local_dir=self._base_model_path,
            )
            logger.info("Base model downloaded successfully")
        else:
            logger.info(
                f"Base model already exists at {self._base_model_path}, skipping download"
            )

    def _download_adapter_model(self, force_download: bool = False):
        self._adapter_model_path = Path(self.settings.nuclia_model_cache) / "REMi-v0"
        if force_download or not self._adapter_model_path.exists():
            logger.info(
                f"Downloading adapter model to {self._adapter_model_path}, to override this behavior, please provide the path in the settings or as an environment variable `NUCLIA_EVAL_MODEL_CACHE`"
            )
            snapshot_download(
                repo_id="nuclia/REMi-v0",
                local_dir=self._adapter_model_path,
            )
            logger.info("Adapter model downloaded successfully")
        else:
            logger.info(
                f"Adapter model already exists at {self._adapter_model_path}, skipping download"
            )

    def evaluate_rag(
        self, query: str, answer: str, contexts: list[str]
    ) -> Tuple[AnswerRelevanceResponse, list[GroundednessCtxRelevanceResponse]]:
        # Answer relevance
        answer_relevance = self.answer_relevance(query, answer)
        # Groundedness and context relevance
        groundedness_ctx_relevances = self.groundedness_ctx_relevance(
            query, answer, contexts
        )
        return answer_relevance, groundedness_ctx_relevances

    def answer_relevance(self, query: str, answer: str) -> AnswerRelevanceResponse:
        system_message = self._get_system_message()
        answer_relevance_message = self._get_answer_relevance_message(query, answer)
        response = self._chat_completion_request(
            [system_message, answer_relevance_message],
            AnswerRelevanceResponse,  # type: ignore
        )
        return response

    def groundedness_ctx_relevance(
        self, query: str, answer: str, contexts: list[str]
    ) -> list[GroundednessCtxRelevanceResponse]:
        system_message = self._get_system_message()
        groundedness_ctx_rel_messages = self._get_groundedness_ctx_rel_messages(
            query, answer, contexts
        )
        responses = []
        for message in groundedness_ctx_rel_messages:
            responses.append(
                self._chat_completion_request(
                    [system_message, message],
                    GroundednessCtxRelevanceResponse,  # type: ignore
                )
            )
        return responses

    def _chat_completion_request(
        self, messages: list[ChatMessageType], target_model: Type[T]
    ) -> T:
        request = ChatCompletionRequest(
            messages=messages,
            tools=self._get_tools(),
            tool_choice=ToolChoice.any,  # type: ignore
        )
        tokens = self.tokenizer.encode_chat_completion(request).tokens
        out_tokens, _ = generate(
            [tokens],
            self.model,
            max_tokens=512,
            temperature=0.0,
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )
        response = self._validate_generation(out_tokens, target_model)
        return response

    def _validate_generation(
        self, out_tokens: list[list[int]], target_model: Type[T]
    ) -> T:
        if not out_tokens:
            raise NoOutputException("No output generated")
        # First token must be token call
        if out_tokens[0][0] != 5:
            raise InvalidToolCallException("First token is not a tool call")

        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        try:
            calls = json.loads(result)
            call = FunctionCall.model_validate(calls[0])
            model = target_model.model_validate_json(call.arguments)
        except (TypeError, ValueError, KeyError, ValidationError):
            raise InvalidToolCallException("Could not parse response")
        return model

    def _get_tools(self) -> list[Tool]:
        if self._tools is None:
            self._tools = [
                Tool.model_validate(tool)
                for tool in [ANSWER_RELEVANCE_TOOL, GROUNDEDNESS_CTX_RELEVANCE_TOOL]
            ]
        return self._tools

    def _get_system_message(self) -> SystemMessage:
        if self._system_message is None:
            self._system_message = SystemMessage(
                content="You are an AI specialized in computing metrics for evaluating Retrieval Augmented Generation (RAG) experiences, use the tools at your disposal to report each of the metrics requested by the user.",
            )
        return self._system_message

    def _get_answer_relevance_message(self, query: str, answer: str) -> UserMessage:
        return UserMessage(
            content=ANSWER_RELEVANCE_TEMPLATE.format(query=query, answer=answer),
        )

    def _get_groundedness_ctx_rel_messages(
        self, query: str, answer: str, contexts: list[str]
    ) -> list[UserMessage]:
        return [
            UserMessage(
                content=GROUNDEDNESS_CTX_RELEVANCE_TEMPLATE.format(
                    query=query, answer=answer, context=context
                ),
            )
            for context in contexts
        ]
