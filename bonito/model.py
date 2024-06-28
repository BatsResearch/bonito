from datasets import Dataset
from vllm import LLM, SamplingParams
from .abstract import AbstractBonito
from copy import deepcopy


class Bonito(LLM, AbstractBonito):
    def generate_tasks(
        self,
        text_dataset: Dataset,
        context_col: str,
        task_type: str,
        sampling_params: SamplingParams,
        greedy_output: bool = False,
        **kwargs,
    ) -> Dataset:
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
            greedy_output (bool): Indicates whether to use
                greedy decoding only for output generation. False
                by default.

        Returns:
            Dataset: The synthetic dataset with the generated tasks.
        """
        processed_dataset = self._prepare_bonito_input(
            text_dataset, task_type, context_col, **kwargs
        )

        if greedy_output:
            generated_dataset = self.generate_with_greedy_output(
                processed_dataset, sampling_params
            )
        else:
            generated_dataset = self.generate_simple(processed_dataset, sampling_params)

        # convert to dataset

        # filter out the examples that cannot be parsed
        synthetic_dataset = self._postprocess_dataset(
            generated_dataset, context_col="context", **kwargs
        )

        return synthetic_dataset

    def generate_simple(
        self, processed_dataset: Dataset, sampling_params: SamplingParams
    ) -> Dataset:
        """
        Generates tasks using the Bonito model.

        Args:
            processed_dataset (Dataset): The formatted and preprocessed
                input dataset.
            sampling_params (SamplingParams): The parameters for sampling.
        """
        outputs = self.generate(processed_dataset["input"], sampling_params)

        # collect multiple generations into one dataset object
        examples = []
        for i, example in enumerate(processed_dataset.to_list()):
            for output in outputs[i].outputs:
                examples.append(
                    {"context": example["context"], "prediction": output.text.strip()}
                )

        synthetic_dataset = Dataset.from_list(examples)

        return synthetic_dataset

    def generate_with_greedy_output(
        self, processed_dataset: Dataset, sampling_params: SamplingParams
    ) -> Dataset:
        """
        Generates tasks using the Bonito model and use greedy decoding to
        generate the output.

        Args:
            processed_dataset (Dataset): The formatted and preprocessed
                input dataset.
            sampling_params (SamplingParams): The parameters for sampling.
        """

        # copy_sampling_params = deepcopy(sampling_params)
        # copy_sampling_params.stop = ["\n<|pipe|>\n"]
        sampling_params.stop = ["\n<|pipe|>\n"]
        preds_with_inputs = self.generate(processed_dataset["input"], sampling_params)

        prompts_with_inputs = []
        input_context = processed_dataset["context"]
        contexts = []
        inputs = []
        for i in range(len(preds_with_inputs)):
            prompt = preds_with_inputs[i].prompt
            for output in preds_with_inputs[i].outputs:
                inputs.append(f"{output.text}\n<|pipe|>\n")
                prompts_with_inputs.append(f"{prompt}{output.text}\n<|pipe|>\n")
                contexts.append(input_context[i])

        # change the temperature to 0 and remove the stop token
        copy_sampling_params = deepcopy(sampling_params)
        copy_sampling_params.temperature = 0
        copy_sampling_params.stop = []
        copy_sampling_params.num_samples = 1

        # generate the rest of the output
        _outputs = self.generate(prompts_with_inputs, copy_sampling_params)
        examples = []
        for i in range(len(_outputs)):
            examples.append(
                {
                    "context": contexts[i],
                    "prediction": inputs[i] + _outputs[i].outputs[0].text,
                }
            )

        generated_dataset = Dataset.from_list(examples)

        return generated_dataset
