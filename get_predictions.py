from llm import GPT
import json
import re
import random
import argparse
from tqdm import tqdm

def prompt_to_chatml(
    prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"
):
    """Convert a text prompt to ChatML format

    Examples
    --------
    >>> prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>system
    name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\nWho's
    there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> prompt_to_chatml(prompt)
    [{'role': 'system', 'content': 'You are a helpful assistant.'},
     {'role': 'user', 'content': 'Knock knock.'},
     {'role': 'assistant', 'content': "Who's there?"},
     {'role': 'user', 'content': 'Orange.'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    def string_to_dict(to_convert):
        """Converts a string with equal signs to dictionary. E.g.
        >>> string_to_dict(" name=user university=stanford")
        {'name': 'user', 'university': 'stanford'}
        """
        return {
            s.split("=", 1)[0]: s.split("=", 1)[1]
            for s in to_convert.split(" ")
            if len(s) > 0
        }

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message

def base_pairwise_parse(
    data: dict,
    verbose: bool = False,
) -> tuple[dict, bool]:
    """
    Parse the response from the model.

    Args:
        data: The data to be parsed.
        verbose: Whether to print verbose output.

    Returns:
        tuple[dict, bool]: The parsed response and whether the parsing failed.
    """
    response = data["response"][0]
    text = response["text"]
    match = re.search(r"Output \((\S+)\)", text)
    fail = False
    if match:
        answer = match.group(1)
        if answer == "a":
            result = 1
        elif answer == "b":
            result = 2
        else:
            result = random.randint(1, 2)
            fail = True
            if verbose:
                print(f"Invalid answer {answer}: {text}")
    else:
        fail = True
        result = random.randint(1, 2)
        if verbose:
            print(f"No matching pattern: {text}")
    result = {"winner": result, "tie": 0, "fail": fail}
    return result, fail


def get_predictions(args):
    with open("data/aligneval.gpt-4o-2024-08-06.jsonl") as f:
        data = [json.loads(line) for line in f]
    with open("prompts/pairwise_eval.txt") as f:
        prompt = f.read().strip()
    
    # create model
    model = GPT(
        model_pt=args.model_pt,
        key=args.key_path,
        account=args.account_path,
        parallel_size=args.parallel_size,
        base_url=args.base_url,
    )

    # generate inputs
    inputs = []
    for item in data:
        messages = prompt_to_chatml(prompt)
        messages[-1]["content"] = messages[-1]["content"].format(
            INSTRUCTION=item["instruction"],
            OUTPUT_1=item["output_1"],
            OUTPUT_2=item["output_2"],
        )
        inputs.append(messages)

    with open(f"results/{args.model_pt}.jsonl", "w") as f:
        for i in tqdm(range(0, len(inputs), args.batch_size)):
            batch = inputs[i:i+args.batch_size]
            # generate outputs
            outputs = model.generate(
                batch,
                n=1,
                temperature=0.0,
                max_tokens=32,
            )
            for x, y in zip(batch, outputs):
                y = {"response": y}
                # parse outputs
                parse_result, fail = base_pairwise_parse(y)
                result = {
                    "input": x,
                    "output": y,
                    "winner": parse_result["winner"],
                }
                print(json.dumps(result), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get predictions from a model.")
    parser.add_argument(
        "--model_pt",
        type=str,
        help="The model to use.",
    )
    parser.add_argument(
        "--key_path",
        type=str,
        help="The path to the OpenAI key.",
    )
    parser.add_argument(
        "--account_path",
        type=str,
        help="The path to the OpenAI account.",
    )
    parser.add_argument(
        "--parallel_size",
        type=int,
        default=1,
        help="The number of parallel requests to make.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="The base URL for the OpenAI API.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for generating outputs.",
    )
    
    args = parser.parse_args()

    get_predictions(args)

   
