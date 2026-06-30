"""Grid + seed definitions (experiment-plan.md, "Independent variable").

Single source of truth for the sweep, so run_all just iterates these. The scaffold runs
ONE demo cell; the full grid sweep is the next step.
"""

SEEDS = list(range(10))  # 10 seeds = 10 distinct disparity patterns per parameter combo
MECHANISMS = ["mcar", "mar", "mnar"]
GROUP_GAPS = [0.0, 0.05, 0.10, 0.20, 0.30]
BASE_RATES = [0.05, 0.15, 0.30]  # includes a low-rate cell (Min/Asif/Vaidya: harm even when little)

# Demo cell for the end-to-end smoke run: strong MNAR, strong group-correlation.
DEMO_CELL = dict(mechanism="mnar", base_rate=0.30, group_gap=0.30, seed=0)
