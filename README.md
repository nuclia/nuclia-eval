<!--- BADGES: START --->
[![HF Nuclia](https://img.shields.io/badge/%F0%9F%A4%97-nuclia_HF-yellow)](https://huggingface.co/nuclia)
[![GitHub - License](https://img.shields.io/github/license/nuclia/nuclia-eval?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nuclia-eval?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/nuclia-eval?logo=pypi&style=flat&color=orange)][#pypi-package]
[![Code coverage](https://nuclia.github.io/nuclia-eval/badges/coverage.svg)](https://github.com/nuclia/nuclia-eval/actions)


[#github-license]: https://github.com/nuclia/nuclia-eval/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/nuclia-eval/
<!--- BADGES: END --->

# nuclia-eval
<p align="center">
  <img src="assets/Nuclia_vertical.png" width="350" title="nuclia logo" alt="nuclia, the all-in-one RAG as a service platform.">
</p>

Library for evaluating RAG using Nuclia's models

Its evaluation follows the RAG triad as proposed by [TruLens](https://www.trulens.org/trulens_eval/getting_started/core_concepts/rag_triad/):

![rag triad](assets/RAG_Triad.jpg)

In summary, the metrics **nuclia-eval** provides for a RAG Experience involving a **question** an **answer** and N pieces of **context** are:

* **Answer Relevance**: Answer relevance refers to the directness and appropriateness of the response in addressing the specific question asked, providing accurate, complete, and contextually suitable information.
    * **score**: A score between 0 and 5 indicating the relevance of the answer to the question.
    * **reason**: A string explaining the reason for the score.
* For each of the N pieces of context:
    * **Context Relevance Score**: The context relevance is the relevance of the **context** to the **question**, on a scale of 0 to 5.
    * **Groudedness Score**: Groundedness is defined as the degree of information overlap to which the **answer** contains information that is substantially similar or identical to that in the **context** piece. The score is between 0 and 5.

## Available Models

### REMi-v0

[REMi-v0](https://huggingface.co/nuclia/REMi-v0) (RAG Evaluation MetrIcs) is a LoRa adapter for the 
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) model. 

It has been finetuned by the team at [**nuclia**](nuclia.com) to evaluate the quality of the overall RAG experience.

## Usage

```python
from nuclia_eval import REMi

evaluator = REMiEvaluator()

...
```