import json
import random

DATA_PATH = "data/public.jsonl"
SAMPLE_PATH = "data/sample.jsonl"

data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

mcq = [d for d in data if d.get("options")]
free = [d for d in data if not d.get("options")]

random.shuffle(mcq)
random.shuffle(free)

sample = mcq[0:50]
sample.extend(free[0:50])
random.shuffle(sample)

with open(SAMPLE_PATH, "w") as f:
    for question in sample:
        f.write(json.dumps(question) + "\n")
