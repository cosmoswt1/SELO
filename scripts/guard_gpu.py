#!/usr/bin/env python3
"""GPU-only guard."""

import subprocess
import sys


def main() -> int:
    try:
        smi = subprocess.run(["nvidia-smi"], check=True, text=True, capture_output=True)
        if smi.stdout:
            print("[guard_gpu] nvidia-smi OK")
            print("[guard_gpu] " + smi.stdout.splitlines()[0])
    except Exception as exc:
        print("[GPU 체크 실패] nvidia-smi 실행 실패", file=sys.stderr)
        print(f"에러: {exc}", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  nvidia-smi", file=sys.stderr)
        print("  python -c \"import torch; print(torch.cuda.is_available())\"", file=sys.stderr)
        return 1

    try:
        import torch  # noqa: WPS433
    except Exception as exc:
        print("[GPU 체크 실패] torch import 실패", file=sys.stderr)
        print(f"에러: {exc}", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  python -c \"import torch; print(torch.cuda.is_available())\"", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("[GPU 체크 실패] torch.cuda.is_available() == False", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  nvidia-smi", file=sys.stderr)
        print("  python -c \"import torch; print(torch.cuda.is_available())\"", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
