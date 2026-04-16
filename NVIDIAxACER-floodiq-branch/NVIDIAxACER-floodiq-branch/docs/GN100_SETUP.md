# GN100 Setup

This project now has a validated GN100 path.

## Verified machine state

- OS: Ubuntu on `aarch64`
- Python: `3.12.3`
- GPU: `NVIDIA GB10`
- Driver: `580.142`
- CUDA: `13.0`

## Working environment steps

```bash
cd ~/FloodIQ
uv venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install cudf-cu13 nvidia-physicsnemo
```

## Verified imports

- `cudf`
- `cupy`
- `torch`
- `physicsnemo`

## PhysicsNeMo note

The GN100 `torch 2.11` environment exposed a `physicsnemo` DTensor import mismatch.
FloodIQ now includes a small runtime compatibility shim in
`floodiq/solver_backends/physicsnemo_compat.py` so the PhysicsNeMo FNO import works in this setup.

## Verified FloodIQ run

The real-data path was validated on the GN100 with:

- `cudf=True`
- `data_mode=real`
- `25,000` sewer complaint rows loaded
- `30,000` elevation points loaded

The PhysicsNeMo surrogate backend was also validated after training a checkpoint on the GN100.

Sample heavy-rain output on the GN100:

- `3.15 in/hr` for `1.0 hr`
- `538` flooded cells
- `1.06 m` peak depth

## Current scale-up path

- the PhysicsNeMo backend can now load checkpoint-specific architecture metadata
- the trainer supports larger GN100-heavy runs:
  - larger grids
  - larger sample counts
  - larger FNO models
  - bigger NYC data limits

See [GN100_TRAINING_PLAN.md](/Users/jnanasreekonda/PycharmProjects/FloodIQ/docs/GN100_TRAINING_PLAN.md) for the recommended heavy training commands.
