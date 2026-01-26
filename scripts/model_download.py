from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="Supertone/supertonic-2",
    local_dir="./assets",
    local_dir_use_symlinks=False,
)
print(local_path)
