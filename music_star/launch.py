"""Distributed training launcher for Music-STAR."""

from __future__ import annotations

import subprocess
import sys
import time

import torch

from music_star.utils import free_port


def main():
    """Launch distributed ``music_star`` training across visible CUDA devices.

    Raises
    ------
    RuntimeError
        If no CUDA devices are available.
    SystemExit
        If any worker exits with a non-zero status.
    """

    args = sys.argv[1:]
    gpus = torch.cuda.device_count()
    if gpus == 0:
        raise RuntimeError("No CUDA devices available for distributed training.")

    port = free_port()
    args += ["--world-size", str(gpus), "--master", f"127.0.0.1:{port}"]
    tasks = []

    for gpu in range(gpus):
        kwargs = {}
        if gpu > 0:
            kwargs["stdin"] = subprocess.DEVNULL
            kwargs["stdout"] = subprocess.DEVNULL
        task = subprocess.Popen(
            [sys.executable, "-m", "music_star", *args, "--rank", str(gpu)],
            **kwargs,
        )
        task.rank = gpu
        tasks.append(task)

    failed = False
    try:
        while tasks:
            for task in list(tasks):
                try:
                    exitcode = task.wait(0.1)
                except subprocess.TimeoutExpired:
                    continue
                tasks.remove(task)
                if exitcode:
                    print(f"Task {task.rank} died with exit code {exitcode}", file=sys.stderr)
                    failed = True
            if failed:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        for task in tasks:
            task.terminate()
        raise

    if failed:
        for task in tasks:
            task.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
