#!/usr/bin/env python3
"""
Environment setup script for Story Reader pipeline.
Automatically loads API keys from text files into environment variables
and prepares automatic sourcing for subprocesses.
"""

import os
import sys
from pathlib import Path
import platform

EXPORT_FILE = "env_exports.sh"  # shell script to source in other scripts

def load_api_keys():
    """Load API keys from text files into environment variables and export file."""

    project_root = Path(__file__).parent

    hf_file = project_root / "HF_TOKEN.txt"
    pexels_file = project_root / "pexels.txt"

    loaded_keys = {}

    def read_key(file_path, env_name):
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            os.environ[env_name] = line
                            loaded_keys[env_name] = line
                            print(f"✓ Loaded {env_name} from {file_path}")
                            return True
                print(f"⚠ Warning: {file_path} contains no valid token")
            except Exception as e:
                print(f"✗ Error reading {file_path}: {e}")
        else:
            print(f"⚠ Warning: {file_path} not found")
        return False

    hf_ok = read_key(hf_file, "HF_TOKEN")
    pexels_ok = read_key(pexels_file, "PEXELS_API_KEY")

    # Write export file so subprocesses can source it
    if loaded_keys:
        export_path = project_root / EXPORT_FILE
        with open(export_path, "w") as f:
            for k, v in loaded_keys.items():
                f.write(f'export {k}="{v}"\n')
        print(f"\n✓ Created export file for subprocesses: {export_path}")

        if platform.system() != "Windows":
            # For Unix shells, print command to auto source
            print(f"\nTo automatically import variables in this shell, run:")
            print(f"eval \"$(cat {EXPORT_FILE})\"")
        else:
            print("\nOn Windows, manually set environment variables in PowerShell or CMD from the export file.")

    # Status summary
    print("\n" + "=" * 50)
    print("API KEY STATUS:")
    print(f"HF_TOKEN: {'✓ Set' if hf_ok else '✗ Not set'}")
    print(f"PEXELS_API_KEY: {'✓ Set' if pexels_ok else '✗ Not set'}")
    print("=" * 50)

    if not (hf_ok and pexels_ok):
        print("\nTo set up API keys:")
        print("1. Create HF_TOKEN.txt with your Hugging Face token")
        print("2. Create pexels.txt with your Pexels API key")
        print("3. Run this script again or eval the export file")
        return False

    return True


def create_sample_files():
    """Create sample API key files with instructions."""
    project_root = Path(__file__).parent

    hf_sample = project_root / "HF_TOKEN.txt.sample"
    if not hf_sample.exists():
        hf_sample.write_text("""# Hugging Face API Token
# To get your token:
# 1. Go to https://huggingface.co/
# 2. Sign in and go to Settings → Access Tokens
# 3. Click "New token" and copy it here
your_hf_token_here
""")
        print(f"Created sample file: {hf_sample}")

    pexels_sample = project_root / "pexels.txt.sample"
    if not pexels_sample.exists():
        pexels_sample.write_text("""# Pexels API Key
# To get your API key:
# 1. Go to https://www.pexels.com/api/
# 2. Sign up and create an API key
your_pexels_api_key_here
""")
        print(f"Created sample file: {pexels_sample}")

    print("\nSample files created. Copy them to:")
    print("- HF_TOKEN.txt (remove .sample extension)")
    print("- pexels.txt (remove .sample extension)")
    print("Then replace placeholders with your actual API keys.")


def main():
    """Main function."""
    print("Story Reader Pipeline - Environment Setup")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == "--create-samples":
        create_sample_files()
        return

    success = load_api_keys()

    if success:
        print("\n✓ Environment setup complete! You can now run the pipeline.")
        if platform.system() != "Windows":
            print(f"Auto-import variables with:\neval \"$(cat {EXPORT_FILE})\"")
        else:
            print("On Windows, manually source the export file or set environment variables in PowerShell.")
        print("Example:")
        print("python -m story_reader --input narration.wav --use-pexels --upscale")
    else:
        print("\n✗ Environment setup incomplete. Use --create-samples to generate sample files:")
        print("python setup_env.py --create-samples")


if __name__ == "__main__":
    main()
