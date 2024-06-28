from bonito import Bonito
from datasets import load_dataset
from vllm import SamplingParams
from transformers import set_seed

# load the model
bonito = Bonito("BatsResearch/bonito-v1")

# load the dataset
unannotated_dataset = load_dataset(
    "BatsResearch/bonito-experiment", "unannotated_pubmed_qa"
)["train"].select(range(10))


def test_greedy_output_with_greedy():
    """
    Test the generate_tasks method with greedy output generation.
    The greedy decoding should be the same with greedy_output=True
    and greedy_output=False.
    """
    # load the dataset
    unannotated_dataset = load_dataset(
        "BatsResearch/bonito-experiment", "unannotated_pubmed_qa"
    )["train"].select(range(10))

    sampling_params = SamplingParams(max_tokens=256, top_p=0.95, temperature=0.0, n=1)
    synthetic_dataset_simple = bonito.generate_tasks(
        unannotated_dataset,
        context_col="input",
        task_type="ynqa",
        sampling_params=sampling_params,
        greedy_output=False,
    )

    synthetic_dataset_greedy = bonito.generate_tasks(
        unannotated_dataset,
        context_col="input",
        task_type="ynqa",
        sampling_params=sampling_params,
        greedy_output=True,
    )

    # check if the outputs are the same
    synthetic_dataset_greedy = synthetic_dataset_greedy.to_dict()
    synthetic_dataset_simple = synthetic_dataset_simple.to_dict()
    for i in range(len(synthetic_dataset_simple["input"])):
        assert (
            synthetic_dataset_simple["input"][i] == synthetic_dataset_greedy["input"][i]
        )
        assert (
            synthetic_dataset_simple["output"][i]
            == synthetic_dataset_greedy["output"][i]
        )


def test_greedy_output_with_sampling():
    """
    Test the generate_tasks method with greedy output generation.
    The greedy decoding should be the same with greedy_output=True
    and greedy_output=False.
    """
    # increased the number of samples to 5
    sampling_params = SamplingParams(max_tokens=256, top_p=0.95, temperature=1.0, n=10)
    set_seed(42)
    synthetic_dataset_simple = bonito.generate_tasks(
        unannotated_dataset,
        context_col="input",
        task_type="ynqa",
        sampling_params=sampling_params,
        greedy_output=False,
    )

    sampling_params = SamplingParams(max_tokens=256, top_p=0.95, temperature=1.0, n=10)
    set_seed(42)
    synthetic_dataset_greedy = bonito.generate_tasks(
        unannotated_dataset,
        context_col="input",
        task_type="ynqa",
        sampling_params=sampling_params,
        greedy_output=True,
    )

    # check if the outputs are the same
    synthetic_dataset_greedy = synthetic_dataset_greedy.to_dict()
    synthetic_dataset_simple = synthetic_dataset_simple.to_dict()
    assert len(synthetic_dataset_greedy["input"]) == len(
        synthetic_dataset_simple["input"]
    )
    for i in range(len(synthetic_dataset_greedy["input"])):
        assert (
            synthetic_dataset_simple["input"][i] == synthetic_dataset_greedy["input"][i]
        )
