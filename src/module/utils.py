import datasets
import warnings
import langid
import torch
from transformers import AutoTokenizer, AutoModel

def load_corpus(corpus_path: str):
    print("Begin Loading Corpus:")
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=8
    )
    print("Finish Loading Corpus!")
    return corpus


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def judge_zh(input_str: str):
    assert isinstance(input_str, str), input_str
    if len(input_str) == 0:
        return False
    detect_result = langid.classify(input_str)
    if detect_result[0] == 'zh':
        return True
    else:
        return False

def set_default_instruction(model_name, is_query=True, is_zh=False):
    instruction = ""
    if "e5" in model_name.lower():
        if is_query:
            instruction = "query: "
        else:
            instruction = "passage: "

    if "bge" in model_name.lower():
        if is_query:
            if "zh" in model_name.lower() or is_zh:
                instruction = "为这个句子生成表示以用于检索相关文章："
            else:
                instruction = "Represent this sentence for searching relevant passages: "

    return instruction

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    elif pooling_method == "last":
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            return last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), sequence_lengths]
    else:
        raise NotImplementedError("Pooling method not implemented!")

def load_model(
        model_path: str, 
        use_fp16: bool = False
    ):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer


def resolve_max_tokens(params: dict, generation_params: dict, prioritize_new_tokens: bool = False) -> dict:
    """
    Resolve and validate max_tokens parameters from both params and generation_params.

    Args:
        params: Dictionary containing user-provided parameters
        generation_params: Dictionary containing generation-specific parameters
        prioritize_new_tokens: If True, max_new_tokens takes precedence over max_tokens
                             If False, max_tokens takes precedence (default behavior)

    Returns:
        Updated generation_params dictionary
    """

    def get_token_params(param_dict: dict) -> tuple:
        """Extract max_tokens and max_new_tokens from a parameter dictionary."""
        return (param_dict.pop("max_tokens", None), param_dict.pop("max_new_tokens", None))

    def resolve_tokens(max_tokens: int, max_new_tokens: int) -> int:
        """
        Resolve between max_tokens and max_new_tokens values based on priority.
        Returns the resolved token value or None if no valid value found.
        """
        # If either value is None, return the non-None value
        if max_tokens is None:
            return max_new_tokens
        if max_new_tokens is None:
            return max_tokens

        # Both values exist but are different
        if max_tokens != max_new_tokens:
            if prioritize_new_tokens:
                warnings.warn(
                    f"max_tokens ({max_tokens}) and max_new_tokens ({max_new_tokens}) "
                    f"are different. Using max_new_tokens value as it has priority."
                )
                return max_new_tokens
            else:
                warnings.warn(
                    f"max_tokens ({max_tokens}) and max_new_tokens ({max_new_tokens}) "
                    f"are different. Using max_tokens value as it has priority."
                )
                return max_tokens

        # Both values are equal
        return max_tokens

    # Try to resolve from params first, then fall back to generation_params
    max_tokens, max_new_tokens = get_token_params(params)
    final_max_tokens = resolve_tokens(max_tokens, max_new_tokens)

    # If no valid tokens found in params, try generation_params
    if final_max_tokens is None:
        max_tokens, max_new_tokens = get_token_params(generation_params)
        final_max_tokens = resolve_tokens(max_tokens, max_new_tokens)

    generation_params.pop("max_new_tokens", None)
    generation_params.pop("max_tokens", None)
    if final_max_tokens is not None:
        if prioritize_new_tokens:
            generation_params["max_new_tokens"] = final_max_tokens
        else:
            generation_params["max_tokens"] = final_max_tokens
    return generation_params