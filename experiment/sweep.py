"""Full grid sweep: every (mechanism, base_rate, group_gap, seed) cell through run_cell.

Each cell yields one (statistic, EO gap, DP gap) observation. The product of the grid in
configs/grid.py is 3 x 3 x 5 x 5 = 225 cells.
"""
from __future__ import annotations

import itertools

import pandas as pd

from experiment.configs.grid import BASE_RATES, GROUP_GAPS, MECHANISMS, SEEDS
from src.pipeline.run_cell import run_cell


def run_grid(X, y, group, inject_columns, mar_driver, progress=True) -> pd.DataFrame:
    combos = list(itertools.product(MECHANISMS, BASE_RATES, GROUP_GAPS, SEEDS))
    rows = []
    for i, (mech, br, gg, seed) in enumerate(combos, 1):
        rows.append(run_cell(X, y, group, inject_columns=inject_columns, mar_driver=mar_driver,
                             mechanism=mech, base_rate=br, group_gap=gg, seed=seed))
        if progress and (i % 25 == 0 or i == len(combos)):
            print(f"  swept {i}/{len(combos)} cells")
    return pd.DataFrame(rows)
