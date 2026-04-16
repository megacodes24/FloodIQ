from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeCapabilities:
    cudf: bool
    cuspatial: bool
    physicsnemo: bool
    torch_cuda: bool


def detect_runtime_capabilities() -> RuntimeCapabilities:
    torch_cuda = False
    try:
        import torch  # type: ignore

        torch_cuda = bool(torch.cuda.is_available())
    except Exception:
        torch_cuda = False

    return RuntimeCapabilities(
        cudf=importlib.util.find_spec("cudf") is not None,
        cuspatial=importlib.util.find_spec("cuspatial") is not None,
        physicsnemo=importlib.util.find_spec("physicsnemo") is not None,
        torch_cuda=torch_cuda,
    )
