"""Download FreshRetailNet-50K (about 4.5 million rows) into data/raw."""
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Dingdong-Inc/FreshRetailNet-50K",
    repo_type="dataset",
    local_dir="data/raw",
)
print("Saved to data/raw")
