from __future__ import annotations

import numpy as np

from .models import GridData


def load_demo_grid(size: int = 48, block_span: int = 8, seed: int = 11) -> GridData:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]

    base_slope = 7.0 - (0.045 * x + 0.07 * y)
    channel = np.exp(-((x - size * 0.62) ** 2) / (2 * (size * 0.08) ** 2)) * 0.9
    basin = np.exp(-(((x - size * 0.38) ** 2) + ((y - size * 0.28) ** 2)) / (2 * (size * 0.09) ** 2)) * 1.3
    roughness = rng.normal(0.0, 0.08, size=(size, size))
    elevation = np.clip(base_slope - channel - basin + roughness, 0.0, None)

    infiltration_rate = np.full((size, size), 1.1e-6, dtype=float)
    park_band = (x < size * 0.18) & (y > size * 0.50)
    infiltration_rate[park_band] = 4.2e-6
    low_permeability = (x > size * 0.45) & (y > size * 0.35)
    infiltration_rate[low_permeability] = 5.5e-7

    building_mask = ((x % block_span) < 2) & ((y % block_span) < 6)
    avenue_mask = (x % block_span == 0) | (y % block_span == 0)
    building_mask &= ~avenue_mask

    drain_capacity = np.zeros((size, size), dtype=float)
    drains = [
        (int(size * 0.17), int(size * 0.84), 2.9e-4),
        (int(size * 0.42), int(size * 0.52), 1.6e-4),
        (int(size * 0.64), int(size * 0.33), 2.2e-4),
        (int(size * 0.76), int(size * 0.72), 2.7e-4),
        (int(size * 0.84), int(size * 0.16), 1.3e-4),
    ]
    for row, col, capacity in drains:
        drain_capacity[max(0, row - 1): row + 2, max(0, col - 1): col + 2] = capacity

    vulnerability = np.clip(
        0.35
        + 0.45 * np.exp(-(((x - size * 0.7) ** 2) + ((y - size * 0.72) ** 2)) / (2 * (size * 0.12) ** 2))
        + 0.25 * np.exp(-(((x - size * 0.22) ** 2) + ((y - size * 0.22) ** 2)) / (2 * (size * 0.1) ** 2)),
        0.0,
        1.0,
    )

    blocks_per_axis = size // block_span
    block_ids = np.full((size, size), -1, dtype=int)
    block_names: list[str] = []
    idx = 0
    for row in range(blocks_per_axis):
        for col in range(blocks_per_axis):
            r0 = row * block_span
            c0 = col * block_span
            block_ids[r0:r0 + block_span, c0:c0 + block_span] = idx
            block_names.append(f"Block-{row + 1}-{col + 1}")
            idx += 1

    return GridData(
        elevation=elevation,
        infiltration_rate=infiltration_rate,
        drain_capacity=drain_capacity,
        building_mask=building_mask,
        vulnerability=vulnerability,
        block_ids=block_ids,
        block_names=block_names,
    )
