#!/usr/bin/env python3
"""Run a training experiment in a persistent Modal Sandbox.

Usage:
    python run_experiment.py > run.log 2>&1

The first invocation creates an H100 sandbox that stays alive for
subsequent runs.  This avoids container cold-starts and preserves
torch.compile caches between experiments.
"""

import os
import sys

import modal

SANDBOX_ID_FILE = ".sandbox_id"
SANDBOX_TIMEOUT = 6 * 3600       # 6-hour max lifetime
SANDBOX_IDLE_TIMEOUT = 30 * 60   # auto-terminate after 30 min idle
EXEC_TIMEOUT = 1800              # 30 min max per training run

CACHE_DIR = "/root/.cache/autoresearch"
CODE_DIR = "/root/autoresearch"
FILES_TO_SYNC = ["train.py", "prepare.py"]

vol = modal.Volume.from_name("autoresearch-cache", create_if_missing=True)

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
)


def _create_sandbox(app):
    """Spin up a fresh H100 sandbox."""
    print("[runner] Creating H100 sandbox …", file=sys.stderr)
    with modal.enable_output():
        sb = modal.Sandbox.create(
            image=image,
            gpu="H100",
            volumes={CACHE_DIR: vol},
            timeout=SANDBOX_TIMEOUT,
            idle_timeout=SANDBOX_IDLE_TIMEOUT,
            app=app,
        )

    sb.mkdir(CODE_DIR, parents=True)

    sb_id = sb.object_id
    with open(SANDBOX_ID_FILE, "w") as f:
        f.write(sb_id)
    print(f"[runner] Sandbox ready: {sb_id}", file=sys.stderr)
    return sb


def get_or_create_sandbox():
    """Reuse an existing sandbox or create a new one."""
    app = modal.App.lookup("autoresearch", create_if_missing=True)

    if os.path.exists(SANDBOX_ID_FILE):
        sb_id = open(SANDBOX_ID_FILE).read().strip()
        try:
            sb = modal.Sandbox.from_id(sb_id)
            if sb.poll() is None:          # still running
                print(f"[runner] Reusing sandbox {sb_id}", file=sys.stderr)
                return sb
            print(f"[runner] Sandbox {sb_id} has exited", file=sys.stderr)
        except Exception as e:
            print(f"[runner] Cannot reconnect ({e})", file=sys.stderr)

    return _create_sandbox(app)


def sync_files(sb):
    """Upload local source files into the sandbox."""
    for fname in FILES_TO_SYNC:
        if not os.path.exists(fname):
            continue
        remote_path = f"{CODE_DIR}/{fname}"
        with open(fname) as local_f:
            content = local_f.read()
        f = sb.open(remote_path, "w")
        f.write(content)
        f.close()
        print(f"[runner] Synced {fname} ({len(content):,} bytes)", file=sys.stderr)


def main():
    sb = get_or_create_sandbox()
    sync_files(sb)

    p = sb.exec(
        "bash", "-c", f"cd {CODE_DIR} && python train.py 2>&1",
        timeout=EXEC_TIMEOUT,
    )

    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    exit_code = p.wait()
    if exit_code != 0:
        print(f"\n[runner] train.py exited with code {exit_code}", file=sys.stderr)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
