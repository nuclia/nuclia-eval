from time import monotonic

from nuclia_eval.models.evaluate import REMiEvaluator


def test_REMi_evaluator():
    # Create an instance of the REMiEvaluator class
    t0 = monotonic()
    evaluator = REMiEvaluator()
    t1 = monotonic()
    query = "What is the Hubble Space Telescope?"
    contexts = [
        "The Hubble Space Telescope is a space telescope that was launched into low Earth orbit in 1990 and remains in operation.",
        "In the solar system, the planets and their moons, comets, asteroids, and meteoroids are all held in orbit by the Sun's gravity.",
        "The size of the elements in the solar system ranges from the Sun, which is the largest, to tiny grains of rock in the asteroid belt.",
        "A microscope is an instrument used to see objects that are too small to be seen by the naked eye. It is used in many scientific fields and is also used in the study of cells and bacteria.",
        "Edwin Hubble was an American astronomer who played a crucial role in establishing the field of extragalactic astronomy and is generally regarded as one of the most important observational cosmologists of the 20th century.",
        "The space bar is a key on a keyboard that is used to insert a space between words or other characters. Studies show that the space bar is one of the most frequently used keys on a keyboard.",
    ]
    answer = "The Hubble Space Telescope (HST) is a space-based observatory that was launched into low Earth orbit by the Space Shuttle Discovery on April 24, 1990. Named after the astronomer Edwin Hubble, it is a project of international cooperation between NASA and the European Space Agency (ESA). The Hubble Space Telescope has made numerous significant contributions to astronomy and cosmology, thanks to its ability to capture high-resolution images and conduct observations without the distortion caused by the Earth's atmosphere."
    out = evaluator.evaluate_rag(query=query, answer=answer, contexts=contexts)
    t2 = monotonic()
    # Print timings
    print(f"Time to load model: {t1 - t0:.2f}s")
    print(f"Time to evaluate: {t2 - t1:.2f}s")

    import pdb

    pdb.set_trace()
    assert out.answer_relevance == 1.0
