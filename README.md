# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Hardware & Inference Time

- **GPU:** NVIDIA L4 (24 GB VRAM) on Google Cloud `g2-standard-12`
- **Approximate total inference time:** ~8 hours 9 minutes for 943 questions (300 MCQ + 643 free-response) on the full private dataset

## Model Weights

This pipeline uses the base model **`Qwen/Qwen3-4B-Thinking-2507`** loaded directly from HuggingFace Hub with INT8 (bitsandbytes) quantization. No fine-tuning was performed, so no custom checkpoint needs to be downloaded. The model is fetched automatically at runtime.

## Running Inference — `run_inference()`

The entry point is `run_inference(data_path, output_path)` in [`run_inference.py`](run_inference.py). It:

1. Loads `Qwen/Qwen3-4B-Thinking-2507` from HuggingFace Hub via vLLM with INT8 quantization.
2. Runs inference on the private dataset, routing MCQ and free-response questions to separate sampling configurations.
3. Outputs the final submission CSV with columns `id` and `response`.

Answer extraction from `\boxed{}` is handled by [`judger.py`](judger.py) at scoring time and is not part of the inference script.

### Setup

```bash
bash twib_team_setup.bash
```

### Reproducing Results

```bash
python run_inference.py <data_path> <output_path>
```

- `<data_path>` — path to the input JSONL file (e.g. `data/private.jsonl`). Each line must be a JSON object with at least `id` and `question` fields; MCQ questions additionally have an `options` list.
- `<output_path>` — path where the submission CSV will be written (e.g. `results/private.csv`). The parent directory is created automatically if it does not exist.

Example:

```bash
python run_inference.py data/private.jsonl results/private.csv
```

This produces a CSV with columns `id` and `response`, sorted by `id`, which is the final submission file.

## Contents

| File | Description |
|---|---|
| `run_inference.py` | **Main inference entry point** — loads model, runs inference, outputs submission CSV |
| `judger.py` | Response scoring logic — extracts `\boxed{}` answers and computes accuracy |
| `utils.py` | Utilities used by `judger.py` |
| `main_experiment.py` | Experimental script used to evaluate different prompt strategies during development |
| `create_sample.py` | Samples a stratified subset of the public dataset for quick local evaluation |
| `twib_team_setup.bash` | One-shot environment setup script (installs uv, creates venv, installs all dependencies) |
| `constraints.txt` | pip dependency constraints (original) |
| `constraints2.txt` | pip dependency constraints used for vLLM-compatible install |
| `experiment_out.txt` | Captured stdout from prompt-strategy experiments |
| `inference_out.txt` | Captured stdout from the final private-set inference run |
| `starter_code_cse151b_comp.ipynb` | Original starter notebook (environment setup and public-set scoring) |
| `baseline_creation.ipynb` | Notebook used to create the baseline submission |
| `no_persona.ipynb` | Experiment notebook — inference without persona prompting |
| `no_persona_aops.ipynb` | Experiment notebook — inference on AoPS questions without persona prompting |
| `aops.ipynb` | Experiment notebook — inference on AoPS-sourced questions |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output CSV files written at runtime |
