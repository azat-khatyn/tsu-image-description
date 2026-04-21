from huggingface_hub import snapshot_download

MODELS = [
    "Salesforce/blip-image-captioning-base",
    "Helsinki-NLP/opus-mt-en-ru",
    "google/siglip-base-patch16-224",
]

for repo_id in MODELS:
    print(f"Downloading: {repo_id}")
    path = snapshot_download(repo_id=repo_id, repo_type="model")
    print(f"Saved to: {path}")

print("Done.")
