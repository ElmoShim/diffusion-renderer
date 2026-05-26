import os
import sys
import argparse
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

COSMOS_MODELS = {
    "forward": {
        "repo_id": "nvidia/Diffusion_Renderer_Forward_Cosmos_7B",
        "local_dir": "Diffusion_Renderer_Forward_Cosmos_7B",
    },
    "inverse": {
        "repo_id": "nvidia/Diffusion_Renderer_Inverse_Cosmos_7B",
        "local_dir": "Diffusion_Renderer_Inverse_Cosmos_7B",
    },
    "tokenizer": {
        "repo_id": "nvidia/Cosmos-Tokenize1-CV8x8x8-720p",
        "local_dir": "Cosmos-Tokenize1-CV8x8x8-720p",
        "gated": True,
    },
}

GATED_HELP = """
{repo_id} is a gated repository.

To download it:
  1. Go to https://huggingface.co/{repo_id}
  2. Accept the license agreement
  3. Make sure your HF token has "Read access to contents of all public gated repos"
     (Settings → Access Tokens → edit token → Permissions)
  4. Run: huggingface-cli login
  5. Re-run this script
"""


def download(repo_id, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    try:
        snapshot_download(repo_id, local_dir=local_dir)
        print(f"Downloaded {repo_id} → {local_dir}")
    except Exception as e:
        msg = str(e)
        chain = e
        while chain.__cause__:
            msg += " " + str(chain.__cause__)
            chain = chain.__cause__
        if "403" in msg or "Forbidden" in msg or "gated" in msg.lower():
            print(GATED_HELP.format(repo_id=repo_id))
            sys.exit(1)
        raise


def main():
    parser = argparse.ArgumentParser(description="Download Cosmos Diffusion Renderer checkpoints from HuggingFace")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Base directory for all checkpoints",
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "forward", "inverse", "tokenizer"],
        help="Which model to download (default: all needed for forward rendering)",
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="Custom HuggingFace repo ID (overrides --model)",
    )
    parser.add_argument(
        "--local_dir", type=str, default=None,
        help="Custom local directory (used with --repo_id)",
    )
    args = parser.parse_args()

    if args.repo_id:
        local_dir = args.local_dir or os.path.join(args.checkpoint_dir, os.path.basename(args.repo_id))
        download(args.repo_id, local_dir)
        return

    if args.model == "all":
        targets = ["forward", "tokenizer"]
    else:
        targets = [args.model]

    for name in targets:
        info = COSMOS_MODELS[name]
        local_dir = os.path.join(args.checkpoint_dir, info["local_dir"])
        download(info["repo_id"], local_dir)

    print("\nDone. Run forward rendering with:")
    print("  python render_zprj.py samples/garment.zprj")


if __name__ == "__main__":
    main()
