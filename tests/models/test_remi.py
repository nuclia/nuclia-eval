import json
import os
from time import monotonic
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from nuclia_eval.exceptions import InvalidToolCallException
from nuclia_eval.models.remi import REMiEvaluator
from nuclia_eval.settings import Settings

MANUAL_TEST = os.getenv("MANUAL_TEST", False)


@pytest.mark.skipif(
    not MANUAL_TEST,
    reason="This test requires a GPU and the downloaded models and is skipped by default.",
)
def test_REMi_evaluator():
    # Create an instance of the REMiEvaluator class
    t0 = monotonic()
    evaluator = REMiEvaluator()
    t1 = monotonic()
    data_path = "/home/learning/oni/llms/rag_metrics_llm/data/miniset_results.json"
    with open(data_path) as f:
        data = json.load(f)
    answer_relevance_MAES = []
    groundedness_MAES = []
    context_relevance_MAES = []
    version = "v0.2"
    for d in data:
        answ_rel = evaluator.answer_relevance(d["question"], d["answer"])
        ctx_rel_responses = evaluator.context_relevance(d["question"], d["contexts"])
        groundedness_responses = evaluator.groundedness(d["answer"], d["contexts"])

        d["answer_relevance_score_" + version] = answ_rel.score
        d["groundedness_scores_" + version] = [
            ctx_rel_grd.score for ctx_rel_grd in groundedness_responses
        ]
        d["context_relevance_scores_" + version] = [
            ctx_rel_grd.score for ctx_rel_grd in ctx_rel_responses
        ]

        d["answer_relevance_MAE_" + version] = abs(
            d["answer_relevance_score_" + version] - d["answer_relevance_score"]
        )

        d["groundedness_MAE_" + version] = sum(
            abs(desired - generated)
            for generated, desired in zip(
                d["groundedness_scores_" + version], d["groundedness_scores"]
            )
        )
        d["groundedness_MAE_" + version] /= len(d["groundedness_scores"])

        d["context_relevance_MAE_" + version] = sum(
            abs(desired - generated)
            for generated, desired in zip(
                d["context_relevance_scores_" + version], d["context_relevance_scores"]
            )
        )
        d["context_relevance_MAE_" + version] /= len(d["context_relevance_scores"])

        answer_relevance_MAES.append(d["answer_relevance_MAE_" + version])
        groundedness_MAES.append(d["groundedness_MAE_" + version])
        context_relevance_MAES.append(d["context_relevance_MAE_" + version])

    print(
        f"Answer Relevance MAE: {sum(answer_relevance_MAES) / len(answer_relevance_MAES)}"
    )
    print(f"Groundedness MAE: {sum(groundedness_MAES) / len(groundedness_MAES)}")
    print(
        f"Context Relevance MAE: {sum(context_relevance_MAES) / len(context_relevance_MAES)}"
    )

    # Save the results
    with open(
        "/home/learning/oni/llms/rag_metrics_llm/data/miniset_results.json", "w"
    ) as f:
        json.dump(data, f)

    import pdb

    pdb.set_trace()

    """query = "What is the Hubble Space Telescope?"
    contexts = [
        "The Hubble Space Telescope is a space telescope that was launched into low Earth orbit in 1990 and remains in operation.",
        "In the solar system, the planets and their moons, comets, asteroids, and meteoroids are all held in orbit by the Sun's gravity.",
        "The size of the elements in the solar system ranges from the Sun, which is the largest, to tiny grains of rock in the asteroid belt.",
        "A microscope is an instrument used to see objects that are too small to be seen by the naked eye. It is used in many scientific fields and is also used in the study of cells and bacteria.",
        "Edwin Hubble was an American astronomer who played a crucial role in establishing the field of extragalactic astronomy and is generally regarded as one of the most important observational cosmologists of the 20th century.",
        "The space bar is a key on a keyboard that is used to insert a space between words or other characters. Studies show that the space bar is one of the most frequently used keys on a keyboard.",
    ]
    answer = "The Hubble Space Telescope (HST) is a space-based observatory that was launched into low Earth orbit by the Space Shuttle Discovery on April 24, 1990. Named after the astronomer Edwin Hubble, it is a project of international cooperation between NASA and the European Space Agency (ESA). The Hubble Space Telescope has made numerous significant contributions to astronomy and cosmology, thanks to its ability to capture high-resolution images and conduct observations without the distortion caused by the Earth's atmosphere."
    out = evaluator.evaluate_rag(query=query, answer=answer, contexts=contexts)"""
    t2 = monotonic()
    # Print timings
    print(f"Time to load model: {t1 - t0:.2f}s")
    print(f"Time to evaluate: {t2 - t1:.2f}s")

    import pdb

    pdb.set_trace()
    # assert out.answer_relevance == 1.0


@patch("nuclia_eval.models.remi.generate")
@patch("nuclia_eval.models.remi.snapshot_download")
@patch("nuclia_eval.models.remi.load_lora_low_mem")
@patch("nuclia_eval.models.remi.Transformer")
@patch("nuclia_eval.models.remi.MistralTokenizer")
def test_REMi_evaluator_mock(
    tokenizer_mock: MagicMock,
    transformer_mock: MagicMock,
    lora_load_mock: MagicMock,
    snapshot_download_mock: MagicMock,
    generate_mock: MagicMock,
):
    # Setup mocks
    snapshot_download_mock.return_value = None
    lora_load_mock.return_value = None
    fake_tokenizer = MagicMock()
    tokenizer_mock.from_file.return_value = fake_tokenizer
    fake_model = MagicMock()
    transformer_mock.from_folder.return_value = fake_model

    # Create custom settings
    settings = Settings(
        nuclia_model_cache="my_cache/",
    )
    # Run code
    evaluator = REMiEvaluator(settings=settings, device="my_device")
    # Check evaluator variables are properly set
    assert evaluator.settings == settings
    # Check that the paths start with the cache path
    assert evaluator._base_model_path.match(settings.nuclia_model_cache + "*")
    assert evaluator._adapter_model_path.match(settings.nuclia_model_cache + "*")

    # Check mock calls
    # 2 Downloads should have been made, one for the base model and one for the adapter model
    snapshot_download_mock.assert_has_calls(
        [
            call(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                allow_patterns=ANY,
                local_dir=evaluator._base_model_path,
            ),
            call(
                repo_id="nuclia/REMi-v0",
                local_dir=evaluator._adapter_model_path,
            ),
        ],
        any_order=True,
    )
    # Check that the tokenizer was loaded
    tokenizer_mock.from_file.assert_called_once()
    # Check that the model was loaded
    transformer_mock.from_folder.assert_called_once()
    # Check that Lora was loaded
    lora_load_mock.assert_called_once()
    # Check that the model was moved to the device
    fake_model.to.assert_called_once_with("my_device")

    # Now load one with default settings
    evaluator = REMiEvaluator()

    # Configure mocks for a rag_evaluation call
    tokenizer_mock.encode_chat_completion.return_value = [1, 2, 3]
    generate_mock.return_value = ([[5, 123, 123]], [[0.1, 0.2, 0.3]])
    fake_tokenizer.instruct_tokenizer.tokenizer.decode.side_effect = [
        # Answer relevance
        '[{"name": "answer_relevance", "arguments": {"reason": "fake", "score": 1}}]',
        # Context relevances
        '[{"name": "context_relevance", "arguments": {"score": 2}}]',
        '[{"name": "context_relevance", "arguments": {"score": 3}}]',
        # Groundedness
        '[{"name": "groundedness", "arguments": {"score": 4}}]',
        '[{"name": "groundedness", "arguments": {"score": 5}}]',
    ]

    # Run evaluation
    answer_relevance, context_relevances, groundednesses = evaluator.evaluate_rag(
        "query", "answer", ["context1", "context2"]
    )
    assert answer_relevance.score == 1 and answer_relevance.reason == "fake"
    assert [cr.score for cr in context_relevances] == [2, 3]
    assert [g.score for g in groundednesses] == [4, 5]

    # Create another evaluator, so that we can check that the model is not downloaded again
    with patch("pathlib.Path.exists", return_value=True):
        evaluator = REMiEvaluator(settings=settings, device="my_device")
    # Check that the download calls were not made again
    assert len(snapshot_download_mock.mock_calls) == 2

    # Check that we raise an error if the first token is not a tool call token
    generate_mock.return_value = ([[123, 123]], [[0.2, 0.3]])
    with pytest.raises(InvalidToolCallException):
        evaluator.evaluate_rag("query", "answer", ["context1", "context2"])

    # Check that we raise an error if the tool call generated is invalid
    generate_mock.return_value = ([[5, 123, 123]], [[0.1, 0.2, 0.3]])
    fake_tokenizer.instruct_tokenizer.tokenizer.decode.side_effect = [
        # Answer relevance with a parameter naming error
        '[{"name": "answer_relevance", "arguments": {"reasoning": "fake", "score": 1}}]',
    ]
    with pytest.raises(InvalidToolCallException):
        evaluator.evaluate_rag("query", "answer", ["context1", "context2"])

    # Check that we raise an error if no output is generated
    generate_mock.return_value = ([], [])
    with pytest.raises(InvalidToolCallException):
        evaluator.evaluate_rag("query", "answer", ["context1", "context2"])

    # Check that we raise an error if the tool name is not the expected one
    generate_mock.return_value = ([[5, 123, 123]], [[0.1, 0.2, 0.3]])
    fake_tokenizer.instruct_tokenizer.tokenizer.decode.side_effect = [
        # Answer relevance but with a different name
        '[{"name": "answer_rel", "arguments": {"reason": "fake", "score": 1}}]',
    ]
    with pytest.raises(InvalidToolCallException):
        evaluator.evaluate_rag("query", "answer", ["context1", "context2"])
