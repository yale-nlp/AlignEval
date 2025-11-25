import json
from sklearn.metrics import cohen_kappa_score
from tabulate import tabulate

BENCHMARK_MODELS = [
    "llama-3-8b",
    "llama-3.1-8b",
    "llama-3.3-70b",
    "qwen-2.5-72b",
    "gpt-4o-mini-2024-07-18",
    "llama-3-70b",
    "llama-3.1-70b",
    "qwen-2-72b",
    "gemma-2-9b",
    "gemma-2-27b",
    "qwen-1.5-32b",
    "qwen-1.5-72b",
    "tulu-2-dpo-70b",
    "gemma-2-2b",
    "gpt-3.5-turbo-0125",
    "gemini-2.0-flash-lite",
    "gpt-4o-2024-05-13",
    "llama-3.1-405b",
    "gemini-1.5-flash",
    "mistral-small-24b",
    "claude-3.5-haiku",
    "claude-3.5-sonnet",
    "gemini-2.0-flash"
]


def main(oracle, models):
    # load oracle data
    with open(f"data/aligneval.{oracle}.jsonl") as f:
        oracle_data = [json.loads(line) for line in f]
    oracle_predictions = [x["winner"] for x in oracle_data]
    model_performance = dict()
    # get model performance
    for model in models:
        with open(f"results/{model}.jsonl") as f:
            model_data = [json.loads(line) for line in f]
        # get model performance
        model_predictions = [x["winner"] for x in model_data]
        # calculate kappa score
        kappa = cohen_kappa_score(oracle_predictions, model_predictions)
        model_performance[model] = kappa
    # sort model performance
    sorted_model_performance = dict(sorted(model_performance.items(), key=lambda item: item[1], reverse=True))
    # print model performance
    print(tabulate(sorted_model_performance.items(), headers=["Model", "Agrement"]))


if __name__ == "__main__":
    # oracle = "gpt-4o-2024-08-06"
    main("gpt-4o-2024-08-06", BENCHMARK_MODELS)
    # oracle = "claude-3.7-sonnet"
    main("claude-3.7-sonnet", BENCHMARK_MODELS)

        
