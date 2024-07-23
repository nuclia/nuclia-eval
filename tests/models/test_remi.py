import os
from time import monotonic
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from nuclia_eval import REMi
from nuclia_eval.exceptions import InvalidToolCallException
from nuclia_eval.settings import Settings

MANUAL_TEST = os.getenv("MANUAL_TEST", False)


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
    evaluator = REMi(settings=settings, device="my_device")
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
    evaluator = REMi()

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
        evaluator = REMi(settings=settings, device="my_device")
    # Check that the download calls were not made again
    # assert len(snapshot_download_mock.mock_calls) == 2

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


@pytest.mark.skipif(
    not MANUAL_TEST,
    reason="This test requires a GPU and the downloaded models and is skipped by default.",
)
def test_REMi_evaluator():
    # Create an instance of the REMiEvaluator class
    t0 = monotonic()
    evaluator = REMi()
    t1 = monotonic()

    query = "By how many Octaves can I shift my OXYGEN PRO 49 keyboard?"
    contexts = [
        """\
* Oxygen Pro 49's keyboard can be shifted 3 octaves down or 4 octaves up.
* Oxygen Pro 61's keyboard can be shifted 3 octaves down or 3 octaves up.

To change the transposition of the keyboard, press and hold Shift, and then use the Key Octave –/+ buttons to lower or raise the keybed by one one, respectively.
The display will temporarily show TRANS and the current transposition (-12 to 12).""",
        """\
To change the octave of the keyboard, use the Key Octave –/+ buttons to lower or raise the octave, respectively
The display will temporarily show OCT and the current octave shift.\n\nOxygen Pro 25's keyboard can be shifted 4 octaves down or 5 octaves up""",
        """\
If your DAW does not automatically configure your Oxygen Pro series keyboard, please follow the setup steps listed in the Oxygen Pro DAW Setup Guides.
To set the keyboard to operate in Preset Mode, press the DAW/Preset Button (on the Oxygen Pro 25) or Preset Button (on the Oxygen Pro 49 and 61).
On the Oxygen Pro 25 the DAW/Preset button LED will be off to show that Preset Mode is selected.
On the Oxygen Pro 49 and 61 the Preset button LED will be lit to show that Preset Mode is selected.""",
    ]
    answer = "Based on the context provided, the Oxygen Pro 49's keyboard can be shifted 3 octaves down or 4 octaves up."

    result = evaluator.evaluate_rag(query=query, answer=answer, contexts=contexts)
    answer_relevance, context_relevances, groundednesses = result
    t2 = monotonic()
    print(
        f"\nAnswer relevance: {answer_relevance.score}, {answer_relevance.reason}"
    )  # 4
    print("Context relevances ", [cr.score for cr in context_relevances])  # [5, 1, 0]
    print("Groundedness: ", [g.score for g in groundednesses])  # [2, 0, 0]
    # Print timings
    print(f"\nTime to load model: {t1 - t0:.2f}s")  # ~15s
    print(f"Time to evaluate: {t2 - t1:.2f}s")  # ~4s

    answer = "Based on the context provided, the Oxygen Pro 61's keyboard can be shifted 4 octaves down or 5 octaves up."
    answer_relevance = evaluator.answer_relevance(query=query, answer=answer)
    context_relevances = evaluator.context_relevance(query=query, contexts=contexts)
    groundednesses = evaluator.groundedness(answer=answer, contexts=contexts)
    t2 = monotonic()
    print(
        f"\nAnswer relevance: {answer_relevance.score}, {answer_relevance.reason}"
    )  # 1
    print("Context relevances ", [cr.score for cr in context_relevances])  # [5, 1, 0]
    print("Groundedness: ", [g.score for g in groundednesses])  # [0, 2, 0]
