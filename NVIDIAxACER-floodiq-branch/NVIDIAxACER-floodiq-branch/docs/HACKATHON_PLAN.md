# FloodIQ Hackathon Plan

## What is already implemented

- Local scenario runner with configurable rainfall intensity and duration
- Grid-based flood simulation over elevation, infiltration, drain, building, and vulnerability layers
- Block-level ranking and plain-language alerts
- Local browser UI for interactive demos

## What to say in the demo right now

1. FloodIQ lets planners run a storm before it happens.
2. We ingest terrain, drainage, permeability, built-environment, and vulnerability signals.
3. Our engine simulates runoff accumulation and drainage at grid-cell resolution, then ranks blocks by impact.
4. The system produces both operational outputs for agencies and explainable alerts for residents.

## Fastest path to the full NVIDIA story

### Phase 1: Data realism

- Replace synthetic terrain with NYC LiDAR DEM tiles
- Map DEP drains and outfalls into the grid
- Attach 311 flooding complaints for validation
- Add NOAA rainfall forecast ingestion

### Phase 2: GPU acceleration

- Use RAPIDS `cuDF` for raster and tabular preprocessing
- Use `cuSpatial` for spatial joins between drains, blocks, and complaints
- Swap the solver backend for PhysicsNeMo/Modulus to run the shallow water equations on GPU

### Phase 3: Planner workflow

- Export the top 20 at-risk blocks
- Generate intervention recommendations:
  - inspect drains
  - pre-stage pumps
  - alert basement residents
  - reroute traffic
- Generate a plain-English emergency summary through a local NIM service

## Judging alignment

- `Technical Execution`: working ingestion -> simulation -> ranking -> UI pipeline
- `Technical Depth`: physics-inspired water redistribution instead of a static map
- `NVIDIA Stack`: architecture is ready for RAPIDS + cuSpatial + PhysicsNeMo/Modulus + NIM
- `Value & Impact`: city agencies can use ranked outputs to prioritize response
- `Frontier Factor`: combines simulation, geospatial analytics, and local AI explanation

## Honest positioning

This repo is the MVP foundation. The current solver is a lightweight fallback so the product is runnable immediately. The winning version should replace the solver backend with PhysicsNeMo/Modulus and calibrate the model against a real event such as Hurricane Ida.
