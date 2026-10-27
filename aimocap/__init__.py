"""aimocap — markerless multi-camera motion capture engine (Python core)."""

from aimocap._gpu import ensure_dlls_on_path

# Ensure CUDA/cuDNN DLLs are loadable BEFORE onnxruntime is imported anywhere.
# This makes the CUDAExecutionProvider available on Windows without manual PATH.
ensure_dlls_on_path()

__version__ = "0.5.0"
