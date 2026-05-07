import json

with open("data/nypl/nypl_train_ready.jsonl") as f:
    sample = json.loads(next(f))

print(sample)
