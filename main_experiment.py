import json
import os

import re
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/sample.jsonl"
OUTPUT_PATH = "results/{}.jsonl"
MAX_TOKENS  = 32768

PROMPT_TYPES = ["baseline", "no_persona", "aops", "no_persona_aops"]
TEMPS = [0.4, 0.6, 0.8]

MATH_SYSTEM_PROMPTS = [
    (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
    ),

    (
    "Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
    ),

    (
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
    ),

    (
    "Solve the problem step-by-step. "
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
]

MCQ_SYSTEM_PROMPTS = [
    (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
    ),

    (
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
    ),

    (
    "You are an expert mathematician. "
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
    ),

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
]

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

def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()

def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

def score_responses(data, responses, judger):
    results = []
    for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold   = item["answer"]

        if is_mcq:
            correct = score_mcq(response, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append({
            "id":       item.get("id"),
            "is_mcq":   is_mcq,
            "gold":     gold,
            "response": response,
            "correct":  correct,
        })

    print(f"Scoring complete. {len(results)} results.")
    
    mcq_res  = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 50)
    return results

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    data = [json.loads(line) for line in open(DATA_PATH)]

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

    # Load Judger for free-form scoring
    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    for prompt_type, math_sys_prompt, mcq_sys_prompt in zip(PROMPT_TYPES, MATH_SYSTEM_PROMPTS, MCQ_SYSTEM_PROMPTS):
        for temp in TEMPS:
            print(f"Running {prompt_type} prompt with {temp} temperature")

            sampling_params = SamplingParams(
                max_tokens=MAX_TOKENS,
                temperature=temp,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
            )

            prompts = build_prompts(math_sys_prompt, mcq_sys_prompt, data, tokenizer)
            print(f"Generating responses for {len(prompts)} questions...")
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            responses = [out.outputs[0].text.strip() for out in outputs]

            results = score_responses(data, responses, judger)

            SAVE_EVAL = True   # Set to False when running on the private test set

            out_path = Path(OUTPUT_PATH.format(f"{prompt_type}-{temp}"))
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w") as f:
                for r in results:
                    if SAVE_EVAL:
                        record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                                "response": r["response"], "correct": r["correct"]}
                    else:
                        record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
                    f.write(json.dumps(record) + "\n")

            print(f"Saved {len(results)} records to {out_path}")
            
if __name__ == "__main__":
    main()
