from datasets import Dataset
from vllm import LLM, SamplingParams
from . import AbstractBonito


class Bonito(LLM, AbstractBonito):
    def generate_tasks(
        self,
        text_dataset: Dataset,
        context_col: str,
        task_type: str,
        sampling_params: SamplingParams,
        **kwargs,
    ):
        """
        Generates tasks using the Bonito model.

        This method takes a text dataset, a context column name,
        a task type, and sampling parameters, and generates tasks
        using the Bonito model. It processes the input dataset,
        generates outputs, collects multiple generations into
        one dataset object, and filters out the examples that
        cannot be parsed.

        Args:
            text_dataset (Dataset): The dataset that provides the text
                for the tasks.
            context_col (str): The name of the column in the dataset
                that provides the context for the tasks.
            task_type (str): The type of the tasks. This can be a
                short form or a full form.
            sampling_params (SamplingParams): The parameters for
                sampling.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset: The synthetic dataset with the generated tasks.
        """
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
