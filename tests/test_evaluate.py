import json
import os
from pathlib import Path
from time import monotonic
from unittest.mock import ANY, MagicMock, call, patch

import pytest

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
    data_path = "/home/learning/oni/llms/rag_metrics_llm/data/miniset.json"
    with open(data_path) as f:
        data = json.load(f)
    answer_relevance_MAES = []
    groundedness_MAES = []
    context_relevance_MAES = []

    for d in data:
        answ_rel, ctx_rel_grd_responses = evaluator.evaluate_rag(
            d["question"], d["answer"], d["contexts"]
        )

        d["answer_relevance_score_v0.1"] = answ_rel.score
        d["groundedness_scores_v0.1"] = [
            ctx_rel_grd.groundedness_score for ctx_rel_grd in ctx_rel_grd_responses
        ]
        d["context_relevance_scores_v0.1"] = [
            ctx_rel_grd.context_relevance_score for ctx_rel_grd in ctx_rel_grd_responses
        ]

        d["answer_relevance_MAE_v0.1"] = abs(
            d["answer_relevance_score_v0.1"] - d["answer_relevance_score"]
        )

        d["groundedness_MAE_v0.1"] = sum(
            abs(desired - generated)
            for generated, desired in zip(
                d["groundedness_scores_v0.1"], d["groundedness_scores"]
            )
        )
        d["groundedness_MAE_v0.1"] /= len(d["groundedness_scores"])

        d["context_relevance_MAE_v0.1"] = sum(
            abs(desired - generated)
            for generated, desired in zip(
                d["context_relevance_scores_v0.1"], d["context_relevance_scores"]
            )
        )
        d["context_relevance_MAE_v0.1"] /= len(d["context_relevance_scores"])

        answer_relevance_MAES.append(d["answer_relevance_MAE_v0.1"])
        groundedness_MAES.append(d["groundedness_MAE_v0.1"])
        context_relevance_MAES.append(d["context_relevance_MAE_v0.1"])

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
    # out = evaluator.groundedness_ctx_relevance(
    #     "What is the Hubble Space Telescope?",
    #     "The Hubble Space Telescope (HST) is a space-based observatory that was launched into low Earth orbit by the Space Shuttle Discovery on April 24, 1990. Named after the astronomer Edwin Hubble, it is a project of international cooperation between NASA and the European Space Agency (ESA). The Hubble Space Telescope has made numerous significant contributions to astronomy and cosmology, thanks to its ability to capture high-resolution images and conduct observations without the distortion caused by the Earth's atmosphere.",
    #     ["The Hubble Space Telescope is a space telescope that was launched into low Earth orbit in 1990 and remains in operation."]
    # )
    out = evaluator.groundedness_ctx_relevance(
        "What are the specs of the OP-1 display?",
        "The specs of the OP-1 display are:\n\n- AMOLED display running at 60 fps\n- 320 x 160 pixel resolution\n- Color Depth: 16.7 M\n- Contrast: 10000:1 (good for outdoor use)\n- Viewing Angle: 170°\n- Life Time: 30,000 h",
        [
            "Display\n\n• AMOLED display running at 60 fps • 320 x 160 pixel resolution • Color Depth: 16.7 M • Contrast: 10000:1 (good for outdoor\n\nuse)\n\n• Viewing Angle: 170° • Life Time: 30,000 h • 1800 mAh li-ion Polymer Battery"
        ],
    )

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
    assert out.answer_relevance == 1.0


@patch("nuclia_eval.models.remi.snapshot_download")
@patch("nuclia_eval.models.remi.load_lora_low_mem")
@patch("nuclia_eval.models.remi.Transformer")
@patch("nuclia_eval.models.remi.MistralTokenizer")
def test_REMi_evaluator_mock(
    tokenizer_mock, transformer_mock, lora_load_mock, snapshot_download_mock
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
    # 2 Download should have been made, one for the base model and one for the adapter model
    import pdb

    pdb.set_trace()
    snapshot_download_mock.assert_called_with(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        allow_patterns=ANY,
        local_dir=evaluator._base_model_path,
    )
    snapshot_download_mock.assert_called_with(
        repo_id="nuclia/REMi-v0",
        local_dir=evaluator._adapter_model_path,
    )
