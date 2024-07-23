<!--- BADGES: START --->
[![Slack](https://img.shields.io/badge/Slack-nuclia-magenta?logo=slack)](https://join.slack.com/t/nuclia-community/shared_invite/zt-2l7jlgi6c-Oohv8j3ygdKOvD_PwZhfdg)
[![HF Nuclia](https://img.shields.io/badge/%F0%9F%A4%97_%20Hugging_Face-nuclia-yellow)](https://huggingface.co/nuclia)
[![GitHub - License](https://img.shields.io/github/license/nuclia/nuclia-eval?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nuclia-eval?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/nuclia-eval?logo=pypi&style=flat&color=orange)][#pypi-package]
[![Code coverage](https://nuclia.github.io/nuclia-eval/badges/coverage.svg)](https://github.com/nuclia/nuclia-eval/actions)


[#github-license]: https://github.com/nuclia/nuclia-eval/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/nuclia-eval/
<!--- BADGES: END --->

# nuclia-eval: Evaluate your RAG with nuclia's models
<p align="center">
  <img src="https://github.com/nuclia/nuclia-eval/blob/main/assets/Nuclia_vertical.png?raw=true" width="350" title="nuclia logo" alt="nuclia, the all-in-one RAG as a service platform.">
</p>

Library for evaluating RAG using **nuclia**'s models

Its evaluation follows the RAG triad as proposed by [TruLens](https://www.trulens.org/trulens_eval/getting_started/core_concepts/rag_triad/):

![rag triad](https://github.com/nuclia/nuclia-eval/blob/main/assets/RAG_Triad.jpg?raw=true)

In summary, the metrics **nuclia-eval** provides for a RAG Experience involving a **question**, an **answer** and N pieces of **context** are:

* **Answer Relevance**: Answer relevance refers to the directness and appropriateness of the response in addressing the specific question asked, providing accurate, complete, and contextually suitable information.
    * **score**: A number between 0 and 5 indicating the score of the relevance of the answer to the question.
    * **reason**: A string explaining the reason for the score.
* For each of the N pieces of context:
    * **Context Relevance Score**: The context relevance is the relevance of the **context** to the **question**, on a scale of 0 to 5.
    * **Groudedness Score**: Groundedness is defined as the degree of information overlap to which the **answer** contains information that is substantially similar or identical to that in the **context** piece. The score is between 0 and 5.

## Installation

```bash
pip install nuclia-eval
```

## Available Models

### REMi-v0

[REMi-v0](https://huggingface.co/nuclia/REMi-v0) (RAG Evaluation MetrIcs) is a LoRa adapter for the 
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) model. 

It has been finetuned by the team at [**nuclia**](nuclia.com) to evaluate the quality of all parts of the RAG experience.

## Usage

```python
from nuclia_eval import REMi

evaluator = REMiEvaluator()

query = "By how many Octaves can I shift my OXYGEN PRO 49 keyboard?"

context1 = """\
* Oxygen Pro 49's keyboard can be shifted 3 octaves down or 4 octaves up.
* Oxygen Pro 61's keyboard can be shifted 3 octaves down or 3 octaves up.

To change the transposition of the keyboard, press and hold Shift, and then use the Key Octave –/+ buttons to lower or raise the keybed by one one, respectively.
The display will temporarily show TRANS and the current transposition (-12 to 12)."""
context2 ="""\
To change the octave of the keyboard, use the Key Octave –/+ buttons to lower or raise the octave, respectively
The display will temporarily show OCT and the current octave shift.\n\nOxygen Pro 25's keyboard can be shifted 4 octaves down or 5 octaves up""",
context3 = """\
If your DAW does not automatically configure your Oxygen Pro series keyboard, please follow the setup steps listed in the Oxygen Pro DAW Setup Guides.
To set the keyboard to operate in Preset Mode, press the DAW/Preset Button (on the Oxygen Pro 25) or Preset Button (on the Oxygen Pro 49 and 61).
On the Oxygen Pro 25 the DAW/Preset button LED will be off to show that Preset Mode is selected.
On the Oxygen Pro 49 and 61 the Preset button LED will be lit to show that Preset Mode is selected.""",

answer = "Based on the context provided, The Oxygen Pro 49's keyboard can be shifted 3 octaves down or 4 octaves up."

result = evaluator.evaluate_rag(query=query, answer=answer, contexts=[context1, context2, context3])
answer_relevance, context_relevances, groundednesses = result

print(f"{answer_relevance.score}, {answer_relevance.reason}")
# 4, The response is relevant to the entire query and answers it completely, but it could be more specific about the limitations of the keyboard.
print([cr.score for cr in context_relevances]) # [5, 1, 0]
print([g.score for g in groundednesses]) # [2, 0, 0]
```
### Granularity

The **REMi** evaluator provides a fine-grained and strict evaluation of the RAG triad. For instance if we slightly modify the answer to the query:

```diff
- answer = "Based on the context provided, The Oxygen Pro 49's keyboard can be shifted 3 octaves down or 4 octaves up."
+ answer = "Based on the context provided, the Oxygen Pro 49's keyboard can be shifted 4 octaves down or 4 octaves up."

...

print([g.score for g in groundednesses]) # [0, 0, 0]
```

As the information provided in the answer is not present in any of the contexts, the groundedness score is 0 for all contexts.

What if the information in the answer does not answer the question?

```diff
- answer = "Based on the context provided, The Oxygen Pro 49's keyboard can be shifted 3 octaves down or 4 octaves up."
+ answer = "Based on the context provided, the Oxygen Pro 61's keyboard can be shifted 3 octaves down or 4 octaves up."

...

print(f"{answer_relevance.score}, {answer_relevance.reason}")
# 1, The response is relevant to the entire query but incorrectly mentions the Oxygen Pro 61 instead of the Oxygen Pro 49
```
### Individual Metrics

We can also compute each metric separately:

```python
...

answer_relevance = evaluator.answer_relevance(query=query, answer=answer)
context_relevances = evaluator.context_relevance(query=query, contexts=[context1, context2, context3])
groundednesses = evaluator.groundedness(answer=answer, contexts=[context1, context2, context3])
...
```

## Feedback and Community

For feedback, questions, or to get in touch with the **nuclia** team, we are available on our [community Slack channel](https://join.slack.com/t/nuclia-community/shared_invite/zt-2l7jlgi6c-Oohv8j3ygdKOvD_PwZhfdg).