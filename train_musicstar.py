import sys
import time
import torch
import subprocess
from musicstar.utils import free_port


def main():
    args = sys.argv[1:]
    gpus = torch.cuda.device_count()
    port = free_port()
    args += ["--world_size", str(gpus), "--master", f"127.0.0.1:{port}"]
    tasks = []

    for gpu in range(gpus):
        kwargs = {}
        if gpu > 0:
            kwargs['stdin'] = subprocess.DEVNULL
            kwargs['stdout'] = subprocess.DEVNULL
            #kwargs['stderr'] = subprocess.DEVNULL
        tasks.append(subprocess.Popen(["python", "-m", "musicstar"] + args + ["--rank", str(gpu)], **kwargs))
        tasks[-1].rank = gpu

    failed = False
    try:
        while tasks:
            for task in tasks:
                try:
                    exitcode = task.wait(0.1)
                except subprocess.TimeoutExpired:
                    continue
                else:
                    tasks.remove(task)
                    if exitcode:
                        print(f"Task {task.rank} died with exit code "
                              f"{exitcode}",
                              file=sys.stderr)
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
