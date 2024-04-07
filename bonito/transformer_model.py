from datasets import Dataset
from tqdm.auto import tqdm
from .abstract import AbstractBonito


class BonitoTransformer(AbstractBonito):
    def __init__(self, model, tokenizer, device="cuda"):
        """
        Construct a Bonito model that uses the Hugging Face Transformers
        library.

        Using the Transformers Hugging Face library we can run the
        Bonito model only using the CPU or run a quantised version.
        Mainly aimed at end users who have limited resources. This
        will be slower than the Bonito class, if possible, it is
        recommended to use that instead.

        Args:
            model: Hugging Face pretrained model
            tokenizer: Hugging Face pretrained tokenizer
            device (str): The device used to do the model inference
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

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

        (Inspired from alexandreteles example code)

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

        # Preprocess the data
        processed_dataset = self._prepare_bonito_input(
            text_dataset, task_type, context_col, **kwargs
        )

        # Collect multiple generations into one dataset object
        examples = []
        for context, prompt in tqdm(
            zip(text_dataset[context_col], processed_dataset["input"]),
            desc="Generating tasks",
            total=len(processed_dataset),
        ):
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(self.device)

            # Run tokenized promt through the model
            outputs = self.model.generate(input_ids, do_sample=True, **sampling_params)

            # Collect up multiple outputs
            for output in outputs:
                examples.append(
                    {
                        "context": context,
                        "prediction": self.tokenizer.decode(
                            output[len(input_ids[0]) :], skip_special_tokens=True
                        ),
                    }
                )

        synthetic_dataset = Dataset.from_list(
            examples
        )  # Need datasets version >= 2.5.0

        # Filter out the examples that cannot be parsed
        synthetic_dataset = self._postprocess_dataset(
            synthetic_dataset, context_col="context", **kwargs
        )

        return synthetic_dataset
