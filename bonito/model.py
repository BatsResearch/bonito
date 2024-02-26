from typing import Optional
from datasets import Dataset
from vllm import LLM
from . import SamplingParams


SHORTFORM_TO_FULL_TASK_TYPES = {
    "exqa": "extractive question answering",
    "mcqa": "multiple-choice question answering",
    "qg": "question generation",
    "qa": "question answering without choices",
    "ynqa": "yes-no question answering",
    "coref": "coreference resolution",
    "paraphrase": "paraphrase generation",
    "paraphrase_id": "paraphrase identification",
    "sent_comp": "sentence completion",
    "sentiment": "sentiment",
    "summarization": "summarization",
    "text_gen": "text generation",
    "topic_class": "topic classification",
    "wsd": "word sense disambiguation",
    "te": "textual entailment",
    "nli": "natural language inference",
}


class Bonito(LLM):
    def __init__(
        self,
        model: str = "BatsResearch/bonito-v1",
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )

    def generate_tasks(
        self,
        text_dataset: Dataset,
        context_col: str,
        task_type: str,
        sampling_params: SamplingParams,
        **kwargs,
    ):
        processed_dataset = self._prepare_bonito_input(
            text_dataset, task_type, context_col, **kwargs
        )
        outputs = self.generate(processed_dataset["input"], sampling_params)

        # collect multiple generations into one dataset object
        examples = []
        for i, example in enumerate(text_dataset.to_list()):
            for output in outputs[i].outputs:
                example["prediction"] = output.text.strip()
                examples.append(example)

        synthetic_dataset = Dataset.from_list(examples)

        # filter out the examples that cannot be parsed
        synthetic_dataset = self._postprocess_dataset(
            synthetic_dataset, context_col, **kwargs
        )

        return synthetic_dataset

    def _prepare_bonito_input(
        self, context_dataset: Dataset, task_type: str, context_col: str, **kwargs
    ) -> Dataset:
        # get the task type name
        if task_type in SHORTFORM_TO_FULL_TASK_TYPES.values():
            full_task_type = task_type
        elif task_type in SHORTFORM_TO_FULL_TASK_TYPES:
            full_task_type = SHORTFORM_TO_FULL_TASK_TYPES[task_type]
        else:
            raise ValueError(f"Task type {task_type} not recognized")

        def process(example):
            input_text = "<|tasktype|>\n" + full_task_type.strip()
            input_text += (
                "\n<|context|>\n" + example[context_col].strip() + "\n<|task|>\n"
            )
            return {
                "input": input_text,
            }

        return context_dataset.map(
            process,
            remove_columns=context_dataset.column_names,
            num_proc=kwargs.get("num_proc", 1),
        )

    def _postprocess_dataset(
        self, synthetic_dataset: Dataset, context_col: str, **kwargs
    ) -> Dataset:
        synthetic_dataset = synthetic_dataset.filter(
            lambda example: len(example["prediction"].split("<|pipe|>")) == 2
        )

        def process(example):
            pair = example["prediction"].split("<|pipe|>")
            context = example[context_col].strip()
            return {
                "input": pair[0].strip().replace("{{context}}", context),
                "output": pair[1].strip(),
            }

        synthetic_dataset = synthetic_dataset.map(
            process,
            remove_columns=synthetic_dataset.column_names,
            num_proc=kwargs.get("num_proc", 1),
        )

        return synthetic_dataset
