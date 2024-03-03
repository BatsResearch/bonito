# Bonito

Bonito is an open-source model for conditional task generation: the task of converting unannotated text into task-specific training datasets for instruction tuning. This repo is a lightweight library for Bonito to easily create synthetic datasets built on top of the Hugging Face `transformers` and `vllm` libraries.

- Paper: [Learning to Generate Instruction Tuning Datasets for
Zero-Shot Task Adaptation](https://arxiv.org/abs/2402.18334)
- Model: [bonito-v1](https://huggingface.co/BatsResearch/bonito-v1)
- Dataset: [ctga-v1](https://huggingface.co/datasets/BatsResearch/ctga-v1)
- Code: To reproduce experiments in our paper, see [nayak-arxiv24-code](https://github.com/BatsResearch/nayak-arxiv24-code).

![Bonito](assets/workflow.png)

## Installation
Create an environment and install the package using the following commands:
```bash
conda create -n bonito python=3.9
conda activate bonito
pip install -e .
```

## Basic Usage
To generate synthetic instruction tuning dataset using Bonito, you can use the following code:
```python
from bonito import Bonito, SamplingParams
from datasets import load_dataset

# Initialize the Bonito model
bonito = Bonito("BatsResearch/bonito-v1")

# load dataset with unannotated text
unannotated_text = load_dataset(
    "BatsResearch/bonito-experiment",
    "unannotated_contract_nli"
)["train"].select(range(10))

# Generate synthetic instruction tuning dataset
sampling_params = SamplingParams(max_tokens=256, top_p=0.95, temperature=0.5, n=1)
synthetic_dataset = bonito.generate_tasks(
    unannotated_text,
    context_col="input",
    task_type="nli",
    sampling_params=sampling_params
)
```

**Supported Task Types [full name (short form)]**: `extractive question answering` (`exqa`), `multiple-choice question answering` (`mcqa`), `question generation` (`qg`), `question answering without choices` (`qa`), `yes-no question answering` (`ynqa`), `coreference resolution` (`coref`), `paraphrase generation` (`paraphrase`), `paraphrase identification` (`paraphrase_id`), `sentence completion` (`sent_comp`), `sentiment` (`sentiment`), `summarization` (`summarization`), `text generation` (`text_gen`), `topic classification` (`topic_class`), `word sense disambiguation` (`wsd`), `textual entailment` (`te`), `natural language inference` (`nli`)

You can use either the full name or the short form to specify the `task_type`.


## Citation
If you use Bonito in your research, please cite the following paper:
```
@article{bonito:arxiv24,
  Author = {Nihal V. Nayak and Yiyang Nan and Avi Trost and Stephen H. Bach},
  Title = {Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation},
  Volume = {arXiv:2402.18334 [cs.CL]},
  Year = {2024}}
```
