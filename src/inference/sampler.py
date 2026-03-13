"""
Batch sample generation using vLLM (phase 1 of two-phase approach).
Generates n samples per problem, saves full outputs for later DTR analysis.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class GeneratedSample:
    problem_id: str
    sample_idx: int
    prompt: str
    prompt_token_ids: List[int]
    generated_text: str
    generated_token_ids: List[int]
    thinking_text: Optional[str] = None
    answer_text: Optional[str] = None


def generate_samples_vllm(
    model_name: str,
    prompts: List[dict],
    n_samples: int = 48,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 32768,
) -> List[List[GeneratedSample]]:
    """
    Generate n_samples per prompt using vLLM.

    Args:
        model_name: HF model name or local path
        prompts: list of dicts with 'id' and 'text' keys
        n_samples: number of samples per prompt
        temperature: sampling temperature
        top_p: nucleus sampling parameter
        max_tokens: maximum new tokens to generate

    Returns:
        List of lists: outer list per prompt, inner list of GeneratedSample
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="bfloat16", max_model_len=max_tokens + 2048)
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    prompt_texts = [p["text"] for p in prompts]
    outputs = llm.generate(prompt_texts, sampling_params)

    all_samples = []
    for prompt_info, output in zip(prompts, outputs):
        prompt_samples = []
        for idx, completion in enumerate(output.outputs):
            generated_text = completion.text
            thinking_text, answer_text = _split_thinking(generated_text)

            sample = GeneratedSample(
                problem_id=prompt_info["id"],
                sample_idx=idx,
                prompt=prompt_info["text"],
                prompt_token_ids=list(output.prompt_token_ids),
                generated_text=generated_text,
                generated_token_ids=list(completion.token_ids),
                thinking_text=thinking_text,
                answer_text=answer_text,
            )
            prompt_samples.append(sample)
        all_samples.append(prompt_samples)

    return all_samples


def _split_thinking(text: str) -> tuple:
    """Split generated text into thinking and answer parts."""
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        return thinking, answer
    return text, ""


def save_samples(samples: List[List[GeneratedSample]], output_dir: str):
    """Save generated samples to JSONL file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "generated_samples.jsonl")
    with open(path, "w") as f:
        for problem_samples in samples:
            for sample in problem_samples:
                f.write(json.dumps(asdict(sample)) + "\n")
    print(f"Saved {sum(len(ps) for ps in samples)} samples to {path}")


def load_samples(path: str) -> List[GeneratedSample]:
    """Load generated samples from JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            samples.append(GeneratedSample(**data))
    return samples
