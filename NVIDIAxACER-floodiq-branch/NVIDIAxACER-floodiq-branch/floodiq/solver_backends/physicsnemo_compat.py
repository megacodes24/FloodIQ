from __future__ import annotations

import sys
import types


def ensure_physicsnemo_torch_compat() -> None:
    """Shim PhysicsNeMo's torch DTensor import for torch versions where the symbol moved."""
    try:
        from torch.distributed.tensor._ops.utils import register_prop_rule  # type: ignore
    except Exception:
        return

    module_name = "torch.distributed.tensor._ops.registration"
    if module_name in sys.modules:
        return

    shim = types.ModuleType(module_name)
    shim.register_prop_rule = register_prop_rule
    sys.modules[module_name] = shim
