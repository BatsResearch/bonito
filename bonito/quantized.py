from typing import List
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import Dataset
from . import AbstractBonito


class QuantizedBonito(AbstractBonito):
    def __init__(self, model_name_or_path):
        self.model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def generate_tasks(
        self,
        text_dataset: Dataset,
        context_col: str,
        task_type: str,
        sampling_params: dict,
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
            sampling_params (dict): The parameters for
                sampling.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset: The synthetic dataset with the generated tasks.
        """
        processed_dataset = self._prepare_bonito_input(
            text_dataset, task_type, context_col, **kwargs
        )

        outputs = self._generate_text(processed_dataset["input"], sampling_params)

        # collect multiple generations into one dataset object
        examples = []
        for i, example in enumerate(text_dataset.to_list()):
            output = outputs[i]
            example["prediction"] = output.strip()
            examples.append(example)

        synthetic_dataset = Dataset.from_list(examples)

        # filter out the examples that cannot be parsed
        synthetic_dataset = self._postprocess_dataset(
            synthetic_dataset, context_col, **kwargs
        )

        return synthetic_dataset

    def _generate_text(
        self,
        dataset: Dataset,
        sampling_params: dict,
        ) -> List[str]:
        """
        Generate text using the model.

        This method takes a dataset of prompts, encodes them,
        generates text using the model, decodes the generated
        text, and appends it to a list.

        Args:
            dataset (Dataset): A dataset containing prompts for text generation.
            sampling_params (dict): Parameters for sampling during generation.

        Returns:
            List[str]: A list of generated texts corresponding to the prompts.
        """
        generated_texts = []

        for prompt in dataset:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.cuda()

            output = self.model.generate(
                input_ids,
                do_sample=True,
                **sampling_params
            )

            generated_text = self.tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
            generated_texts.append(generated_text)

        return generated_texts
