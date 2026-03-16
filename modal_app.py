"""
Modal deployment for autoresearch.

Usage:
    modal run modal_app.py::prepare    # one-time: download data + train tokenizer
    modal run modal_app.py::train      # run a single 5-minute training experiment
"""

import modal

app = modal.App("autoresearch")

# Persistent volume for data + tokenizer (survives between runs)
vol = modal.Volume.from_name("autoresearch-cache", create_if_missing=True)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "kernels>=0.11.7",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
        "numpy>=2.2.6",
        "pyarrow>=21.0.0",
        "requests>=2.32.0",
    )
    .add_local_dir(".", remote_path="/root/autoresearch")
)

CACHE_DIR = "/root/.cache/autoresearch"


@app.function(
    image=image,
    gpu="H100",
    volumes={CACHE_DIR: vol},
    timeout=1800,
)
def prepare():
    """One-time: download data shards + train tokenizer."""
    import subprocess

    result = subprocess.run(
        ["python", "prepare.py"],
        cwd="/root/autoresearch",
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("prepare.py failed")
    vol.commit()


@app.function(
    image=image,
    gpu="H100",
    volumes={CACHE_DIR: vol},
    timeout=1800,  # 30 min: covers torch.compile + 5 min train + eval
)
def train():
    """Run a single 5-minute training experiment."""
    import subprocess

    result = subprocess.run(
        ["python", "train.py"],
        cwd="/root/autoresearch",
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"train.py failed:\n{result.stderr}")
    return result.stdout
