import argparse
import os

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Supertonic model assets from Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        default=os.getenv("MODEL_REPO_ID", "Supertone/supertonic-2"),
        help="Hugging Face model repo id (default: %(default)s)",
    )
    parser.add_argument(
        "--revision",
        default=os.getenv("MODEL_REVISION"),
        help="Model revision (commit SHA or tag). If omitted, latest is used.",
    )
    parser.add_argument(
        "--local-dir",
        default=os.getenv("MODEL_LOCAL_DIR", "./assets"),
        help="Local target directory for downloaded model files (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    local_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
    )
    print(local_path)


if __name__ == "__main__":
    main()
