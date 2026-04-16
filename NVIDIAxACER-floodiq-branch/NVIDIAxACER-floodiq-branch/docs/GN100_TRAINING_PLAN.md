# GN100 Training Plan

This plan is for showing that FloodIQ is using the Acer GN100 for real local compute, not just light inference.

## Goal

Push the GN100 along three axes at once:

1. heavier NYC data ingestion
2. larger Lower Manhattan flood grids
3. larger PhysicsNeMo FNO checkpoints

## Training profiles

### Good demo run

Use this when you want a stronger checkpoint without waiting too long:

```bash
cd ~/FloodIQ
source .venv/bin/activate
FLOODIQ_SOLVER=physicsnemo \
python -m floodiq.train_physicsnemo_surrogate \
  --study-area lower_manhattan \
  --samples 512 \
  --epochs 60 \
  --grid-size 128 \
  --complaint-limit 60000 \
  --elevation-limit 120000 \
  --latent-channels 48 \
  --num-fno-layers 6 \
  --num-fno-modes 16 \
  --decoder-layer-size 96 \
  --learning-rate 0.0008 \
  --output artifacts/physicsnemo_lower_manhattan_heavy.pt
```

### Better GN100 run

Use this when you want a bigger compute story:

```bash
cd ~/FloodIQ
source .venv/bin/activate
FLOODIQ_SOLVER=physicsnemo \
python -m floodiq.train_physicsnemo_surrogate \
  --study-area lower_manhattan \
  --samples 1024 \
  --epochs 80 \
  --grid-size 160 \
  --complaint-limit 90000 \
  --elevation-limit 180000 \
  --latent-channels 64 \
  --num-fno-layers 8 \
  --num-fno-modes 20 \
  --decoder-layer-size 128 \
  --learning-rate 0.0006 \
  --output artifacts/physicsnemo_lower_manhattan_xl.pt
```

### Max hackathon push

Use this only if you have time and want the strongest GN100 compute story:

```bash
cd ~/FloodIQ
source .venv/bin/activate
FLOODIQ_SOLVER=physicsnemo \
python -m floodiq.train_physicsnemo_surrogate \
  --study-area lower_manhattan \
  --samples 1536 \
  --epochs 100 \
  --grid-size 192 \
  --complaint-limit 120000 \
  --elevation-limit 240000 \
  --latent-channels 80 \
  --num-fno-layers 10 \
  --num-fno-modes 24 \
  --decoder-layer-size 160 \
  --learning-rate 0.0005 \
  --output artifacts/physicsnemo_lower_manhattan_max.pt
```

## What to monitor on the GN100

In another terminal:

```bash
watch -n 1 nvidia-smi
```

Capture:

- GPU utilization during training
- training duration
- checkpoint size
- final loss
- grid size used
- number of samples used

## Demo-ready talking points

- We ingest raw NYC Open Data locally on the GN100.
- We train a PhysicsNeMo FNO surrogate locally on large Lower Manhattan grids.
- We scale up both the dataset and the model architecture on-device.
- We use the GN100 for both data preparation and flood-model training, not just API calls.

## Recommended order

1. run the `Good demo run`
2. check that the resulting checkpoint loads in the app
3. if time remains, run the `Better GN100 run`
4. only attempt the `Max hackathon push` if the earlier runs are stable
