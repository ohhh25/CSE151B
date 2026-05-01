# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |


## MY SETUP FOR VLLM COMPATIBILITY
```bash
uv venv .venv2 --seed

source .venv2/bin/activate

uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

uv pip install transformers==4.57.6

uv pip install sympy numpy vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter accelerate -c constraints.txt
```
