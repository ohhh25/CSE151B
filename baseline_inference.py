import json
import os

import re
import sys
import pandas as pd
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/private.csv"
MAX_TOKENS  = 32768

MATH_SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

MCQ_SYSTEM_PROMPT = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

def build_prompt(math_sys_prompt, mcq_sys_prompt, question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return mcq_sys_prompt, f"{question}\n\nOptions:\n{opts_text}"
    return math_sys_prompt, question

def build_prompts(math_sys_prompt, mcq_sys_prompt, data, tokenizer):
    prompts = []
    for item in data:
        system, user = build_prompt(math_sys_prompt, mcq_sys_prompt, item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
            {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
    return prompts

def run_inference(data_path, output_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    data = [json.loads(line) for line in open(data_path)]

    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.50,
        max_model_len=8192,
        trust_remote_code=True,
        max_num_seqs=128,
        max_num_batched_tokens=16384,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    prompts = build_prompts(MATH_SYSTEM_PROMPT, MCQ_SYSTEM_PROMPT, data, tokenizer)
    print(f"Generating responses for {len(prompts)} MCQ questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    responses = [out.outputs[0].text.strip() for out in outputs]
    ids = [item["id"] for item in data]

    df = pd.DataFrame({"id": ids, "response": responses})
    df = df.sort_values("id").reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_inference(sys.argv[1], sys.argv[2])
    else:
        print(f"Usage: python baseline_inference.py <data_path> <output_path>")
