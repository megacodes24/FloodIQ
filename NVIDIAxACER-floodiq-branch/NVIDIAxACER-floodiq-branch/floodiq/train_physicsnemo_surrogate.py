from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .models import RainScenario
from .real_grid import build_real_grid
from .settings import DEFAULT_STUDY_AREA, PROJECT_ROOT, get_study_area
from .simulation import FloodSimulator
from .solver_backends.features import scenario_feature_stack


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PhysicsNeMo FNO surrogate for FloodIQ.")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "physicsnemo_surrogate.pt")
    parser.add_argument("--study-area", default=DEFAULT_STUDY_AREA.slug)
    parser.add_argument("--use-synthetic-grid", action="store_true")
    parser.add_argument("--heavy-rain-bias", action="store_true", default=True)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--complaint-limit", type=int, default=60000)
    parser.add_argument("--elevation-limit", type=int, default=120000)
    parser.add_argument("--latent-channels", type=int, default=48)
    parser.add_argument("--num-fno-layers", type=int, default=6)
    parser.add_argument("--num-fno-modes", type=int, default=16)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layer-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    args = parser.parse_args()

    import torch  # type: ignore

    from .data import load_demo_grid
    from .nyc_open_data import NYCOpenDataClient
    from .providers import detect_runtime_capabilities
    from .solver_backends.physicsnemo_backend import _load_fno_class
    from .solver_backends.physicsnemo_compat import ensure_physicsnemo_torch_compat

    if not torch.cuda.is_available():
        raise RuntimeError("Training the PhysicsNeMo surrogate requires CUDA.")
    ensure_physicsnemo_torch_compat()

    complaint_prior = None
    if args.use_synthetic_grid:
        grid = load_demo_grid()
        selected_area = DEFAULT_STUDY_AREA
    else:
        selected_area = get_study_area(args.study_area)
        bundle = NYCOpenDataClient().fetch_bundle(
            selected_area,
            complaint_limit=args.complaint_limit,
            elevation_limit=args.elevation_limit,
        )
        real_grid_result = build_real_grid(
            bundle.sewer_complaints,
            bundle.elevation_points,
            bundle.street_centerlines,
            bundle.sidewalk_polygons,
            bundle.parking_lot_polygons,
            bundle.catch_basins,
            bundle.outfalls,
            selected_area,
            detect_runtime_capabilities(),
            size=args.grid_size,
        )
        grid = real_grid_result.grid
        complaint_prior = real_grid_result.layers.get("recent_counts")

    features, labels = _build_training_set(grid, args.samples, heavy_rain_bias=args.heavy_rain_bias)
    device = "cuda"
    fno_cls = _load_fno_class()
    model = fno_cls(
        in_channels=features.shape[1],
        out_channels=1,
        dimension=2,
        latent_channels=args.latent_channels,
        num_fno_layers=args.num_fno_layers,
        num_fno_modes=args.num_fno_modes,
        padding=0,
        decoder_layers=args.decoder_layers,
        decoder_layer_size=args.decoder_layer_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    x_train = torch.from_numpy(features).to(device)
    target_scale = float(np.max(labels) or 1.0)
    transformed_labels = np.sqrt(np.clip(labels / target_scale, 0.0, None)).astype(np.float32)
    y_train = torch.from_numpy(transformed_labels).to(device)
    vulnerability_prior = torch.from_numpy(grid.vulnerability.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    if complaint_prior is not None:
        complaint_scale = float(np.max(complaint_prior) or 1.0)
        complaint_prior_tensor = torch.from_numpy((complaint_prior / complaint_scale).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    else:
        complaint_prior_tensor = torch.zeros_like(vulnerability_prior)
    weight = (
        1.0
        + 5.0 * (torch.from_numpy(labels).to(device) > 0.12).float()
        + 1.5 * vulnerability_prior
        + 2.5 * complaint_prior_tensor
    )

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train)
        loss = (((prediction - y_train) ** 2) * weight).mean()
        loss.backward()
        optimizer.step()
        print(f"epoch={epoch + 1} loss={loss.item():.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "samples": args.samples,
            "epochs": args.epochs,
            "study_area": selected_area.slug,
            "target_scale": target_scale,
            "target_transform": "sqrt",
            "grid_shape": list(grid.elevation.shape),
            "complaint_guided": complaint_prior is not None,
            "dataset": {
                "complaint_limit": args.complaint_limit,
                "elevation_limit": args.elevation_limit,
                "use_synthetic_grid": args.use_synthetic_grid,
                "heavy_rain_bias": args.heavy_rain_bias,
            },
            "architecture": {
                "in_channels": int(features.shape[1]),
                "out_channels": 1,
                "dimension": 2,
                "latent_channels": args.latent_channels,
                "num_fno_layers": args.num_fno_layers,
                "num_fno_modes": args.num_fno_modes,
                "padding": 0,
                "decoder_layers": args.decoder_layers,
                "decoder_layer_size": args.decoder_layer_size,
            },
            "optimizer": {
                "name": "Adam",
                "learning_rate": args.learning_rate,
            },
        },
        args.output,
    )
    print(f"saved checkpoint to {args.output}")


def _build_training_set(grid, sample_count: int, heavy_rain_bias: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    simulator = FloodSimulator()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for idx in range(sample_count):
        if heavy_rain_bias:
            rainfall = float(rng.triangular(1.25, 4.2, 6.0))
            duration = float(rng.triangular(0.75, 1.75, 3.5))
        else:
            rainfall = float(rng.uniform(0.75, 5.5))
            duration = float(rng.uniform(0.5, 3.5))
        scenario = RainScenario(
            name=f"surrogate-{idx}",
            rainfall_inches_per_hour=rainfall,
            duration_hours=duration,
        )
        result = simulator.run(grid, scenario)
        features.append(scenario_feature_stack(grid, scenario))
        labels.append(result.max_water_depth_m.astype(np.float32)[None, ...])

    return np.stack(features, axis=0), np.stack(labels, axis=0)


if __name__ == "__main__":
    main()
