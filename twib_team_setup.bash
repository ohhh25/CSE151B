#!/bin/bash

wget -qO- https://astral.sh/uv/install.sh | sh

uv venv .venv2 --seed
source .venv2/bin/activate

uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers==4.57.6
uv pip install sympy numpy vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter accelerate -c constraints2.txt
uv pip install pandas
uv pip install kaggle

python -m ipykernel install --user --name cse151b --display-name "Python (cse151b)"
