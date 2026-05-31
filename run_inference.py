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

MATH_TEMP = 0.6
MCQ_TEMP = 0.8

MATH_SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
    "Example Problem and Response:\n"
    "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$"
    "The $9$ members of a baseball team went to an ice-cream parlor after their game. Each player had a single scoop cone of chocolate, vanilla, or strawberry ice cream. At least one player chose each flavor, and the number of players who chose chocolate was greater than the number of players who chose vanilla, which was greater than the number of players who chose strawberry. Let $N$ be the number of different assignments of flavors to players that meet these conditions. Find the remainder when $N$ is divided by $1000.$ [ANS]\n"
    "We apply casework on the scoops the team gets.\n"
    "Case 1: The scoops are $6,2,1$. Then we have $\\binom{9}{6}\\cdot \\binom{3}{2} = 252$.\n"
    "Case 2: The scoops are $5,3,1$. Then we have $\\binom{9}{5}\\cdot \\binom{4}{3} = 504$.\n"
    "Case 3: The scoops are $4,3,2$. Then we have $\\binom{9}{4}\\cdot \\binom{5}{3} = 1260$.\n"
    "Thus the answer is $252+504+1260=\\boxed{2016}$."
)

MCQ_SYSTEM_PROMPT = (
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
    "Example Problem and Response:\n"
    "Andy and Betsy both live in Mathville. Andy leaves Mathville on his bicycle at $1{:}30$, traveling due north at a steady $8$ miles per hour. Betsy leaves on her bicycle from the same point at $2{:}30$, traveling due east at a steady $12$ miles per hour. At what time will they be exactly the same distance from their common starting point?\n"
    "Options:\n"
    "A. $3{:}30$\n"
    "B. $3{:}45$\n"
    "C. $4{:}00$\n"
    "D. $4{:}15$\n"
    "E. $4{:}30$\n"
    "Andy goes $8$ mph, Betsy goes $12$ mph. We know that Betsy starts 1 hour after Andy, so she will be 8 miles behind.\n"
    "$12x - 8 = 8x$\n"
    "$4x = 8 \\rightarrow x = 2$"
    "2 hours after Betsy's start time is $4{:}30$. Thus the answer is \\boxed{E}."
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

    mcq, free = [], []
    for d in data:
        if d.get("options"):
            mcq.append(d)
        else:
            free.append(d)

    n_mcq, n_free = len(mcq), len(free)
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

    mcq_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=MCQ_TEMP,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    math_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=MATH_TEMP,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    prompts = build_prompts(MATH_SYSTEM_PROMPT, MCQ_SYSTEM_PROMPT, mcq, tokenizer)
    print(f"Generating responses for {n_mcq} MCQ questions...")
    outputs = llm.generate(prompts, sampling_params=mcq_params)
    responses = [out.outputs[0].text.strip() for out in outputs]
    ids = [item["id"] for item in mcq]

    prompts = build_prompts(MATH_SYSTEM_PROMPT, MCQ_SYSTEM_PROMPT, free, tokenizer)
    print(f"Generating responses for {n_free} free response questions...")
    outputs = llm.generate(prompts, sampling_params=math_params)
    responses.extend([out.outputs[0].text.strip() for out in outputs])
    ids.extend([item["id"] for item in free])

    df = pd.DataFrame({"id": ids, "response": responses})
    df = df.sort_values("id").reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_inference(sys.argv[1], sys.argv[2])
    else:
        print(f"Usage: python run_inference.py <data_path> <output_path>")
