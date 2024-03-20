from typing import Dict, Optional, Union
from datasets import Dataset


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


class AbstractBonito:
    def _prepare_bonito_input(
        self, context_dataset: Dataset, task_type: str, context_col: str, **kwargs
    ) -> Dataset:
        """
        Prepares the input for the Bonito model.

        This method takes a context dataset, a task type, and a context
        column name, and prepares the dataset for the Bonito model.
        If the task type is not recognized, it raises a ValueError.

        Args:
            context_dataset (Dataset): The dataset that provides the
                context for the task.
            task_type (str): The type of the task. This can be a
                short form or a full form. If the task type is not
                recognized, a ValueError is raised.
            context_col (str): The name of the column in the dataset
                that provides the context for the task.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset: The prepared dataset for the Bonito model.
        """
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
        """
        Post-processes the synthetic dataset.

        This method takes a synthetic dataset and a context column
        name, and post-processes the dataset. It filters out
        examples where the prediction does not contain exactly two
        parts separated by "<|pipe|>", and then maps each example to a
        new format where the context is inserted into the first part of
        the prediction and the second part of the prediction is used as
        the output.

        Args:
            synthetic_dataset (Dataset): The synthetic dataset to be
                post-processed.
            context_col (str): The name of the column in the dataset
                that provides the context for the tasks.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset: The post-processed synthetic dataset.
        """
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
