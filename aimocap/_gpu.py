"""GPU provider bootstrap for ONNX Runtime on Windows.

ONNX Runtime's CUDA Execution Provider needs the CUDA + cuDNN runtime DLLs to be
loadable at process start. On Windows those aren't on the system PATH by default;
when we install them via pip (`nvidia-cuda-nvrtc-cu12`, `nvidia-cudnn-cu12`,
`nvidia-cublas-cu12`), they land under the venv's `site-packages/nvidia/*/bin`.

This module registers those directories with the Windows DLL loader *before*
onnxruntime is imported, so `CUDAExecutionProvider` shows up automatically.
We use :func:`os.add_dll_directory` (the Win32 ``AddDllDirectory`` API), which is
the reliable mechanism for Python 3.8+ — appending to ``PATH`` alone often fails
for DLLs whose dependencies are resolved at load time.

Call :func:`ensure_dlls_on_path` once at the top of any entrypoint that uses ORT.
On non-Windows or without the nvidia packages, it's a no-op.
"""

from __future__ import annotations

import os
import sys


def _nvidia_dll_dirs() -> list[str]:
    """Return the nvidia/*/bin and torch/lib directories containing CUDA/cuDNN DLLs."""
    dirs: list[str] = []
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            dirs.append(torch_lib)
    except ImportError:
        pass

    try:
        import nvidia  # type: ignore
        roots = list(nvidia.__path__)
    except Exception:
        roots = []

    for root in roots:
        for sub, _dirs, files in os.walk(root):
            if any(f.lower().endswith(".dll") for f in files) and sub.endswith("bin"):
                dirs.append(sub)
    return dirs


def ensure_dlls_on_path(verbose: bool = False) -> bool:
    """Register nvidia CUDA/cuDNN DLL dirs with the loader. Idempotent.

    Returns True if any directories were registered. Must be called BEFORE
    ``import onnxruntime`` for the CUDA EP to register.

    On Windows we use both ``os.add_dll_directory`` (primary, reliable) and
    prepending to ``PATH`` (belt-and-braces, for any loader path that still
    consults it).
    """
    if sys.platform != "win32":
        return False

    dirs = _nvidia_dll_dirs()
    if not dirs:
        if verbose:
            print("[gpu] No nvidia cu12 packages found; CPU-only.")
        return False

    added = False
    for d in dirs:
        # Primary mechanism.
        try:
            os.add_dll_directory(d)
            added = True
            if verbose:
                print(f"[gpu] add_dll_directory({d})")
        except (OSError, FileNotFoundError):
            pass
        # Belt-and-braces: also prepend to PATH.
        path = os.environ.get("PATH", "")
        if d not in path.split(os.pathsep):
            os.environ["PATH"] = d + os.pathsep + path
    return added
