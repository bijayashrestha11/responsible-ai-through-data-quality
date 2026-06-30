"""Demonstrate the missingness-disparity validation gate (the data-engineering showcase).

Shows the gate doing its job at a pre-training validation stage:
  - a batch with NO group-correlated missingness  -> PASS
  - a batch with strong group-correlated missingness -> FAIL (pipeline should halt/alert)
on both Adult and COMPAS. If Great Expectations is installed, also runs the SAME check routed
through GE's validation engine (src/pipeline/integrations.great_expectations_gate) to show the
gate composing on top of a real data-quality framework.

The gate needs only features + the protected-group label — no target, no model — so it runs at a
true ingestion gate. The threshold is a policy knob; tune it from the benchmark detector results.

Usage (from repo root):  python experiment/demo_gate.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from experiment.benchmark.load_adult import load_adult  # noqa: E402
from experiment.benchmark.load_compas import load_compas  # noqa: E402
from src.missingness.inject import inject_missingness  # noqa: E402
from src.pipeline.gate import MissingnessDisparityGate  # noqa: E402

THRESHOLD = 0.10  # fail if D_max (worst-feature group missingness-rate gap) exceeds this


def _batch(X, group, cols, driver, group_gap, seed=0):
    return inject_missingness(X, cols, group, mechanism="mnar", base_rate=0.20,
                              group_gap=group_gap, seed=seed, mar_driver=driver)


def demo(name, loader):
    X, y, group, cols, driver = loader()
    gate = MissingnessDisparityGate(columns=cols, threshold=THRESHOLD)
    clean = _batch(X, group, cols, driver, group_gap=0.0)   # missingness present, but NOT group-correlated
    disparate = _batch(X, group, cols, driver, group_gap=0.30)  # strong group-correlated missingness
    rc, rd = gate(clean, group), gate(disparate, group)
    print(f"\n[{name}] gate threshold = {THRESHOLD}")
    print(f"  no-disparity batch : {rc.message}")
    print(f"  disparate   batch  : {rd.message}")
    assert rc.passed and not rd.passed, "gate should PASS the clean batch and FAIL the disparate one"
    return X, group, cols, disparate


def maybe_ge(disparate, group, cols):
    try:
        import great_expectations  # noqa: F401
    except ImportError:
        print("\n[GE] great_expectations not installed in this env — skipping in-framework run.")
        print("     Verified separately in an isolated venv; see DEVLOG.md.")
        return
    from src.pipeline.integrations import great_expectations_gate
    res = great_expectations_gate(disparate, group, cols, threshold=THRESHOLD)
    print(f"\n[GE] same gate, measured through Great Expectations' engine: {res.message}")


def main():
    print("=== Missingness-disparity validation gate demo ===")
    Xa, ga, colsa, disparate_a = demo("adult", load_adult)
    demo("compas", load_compas)
    maybe_ge(disparate_a, ga, colsa)
    print("\nGate works: it halts a pipeline on group-correlated missingness, before any training.")


if __name__ == "__main__":
    main()
