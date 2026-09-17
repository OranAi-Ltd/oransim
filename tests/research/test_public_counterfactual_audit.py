from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "backend" / "scripts" / "run_public_counterfactual_audit.py"
SPEC = importlib.util.spec_from_file_location("run_public_counterfactual_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_counterfactual_budget_matches_generator_rule() -> None:
    budget = 100.0
    assert budget * MODULE.CF_BUDGET_MULTIPLIER[0] == pytest.approx(60.0)
    assert budget * MODULE.CF_BUDGET_MULTIPLIER[3] == pytest.approx(180.0)


def test_scenario_id_split_is_disjoint_and_exact() -> None:
    frame = pd.DataFrame({"scenario_id": [f"SCN_{i:08d}" for i in range(20)]})
    train, validation, test = MODULE.split_by_scenario_id(frame)
    assert (len(train), len(validation), len(test)) == (16, 2, 2)
    assert set(train.scenario_id).isdisjoint(validation.scenario_id)
    assert set(train.scenario_id).isdisjoint(test.scenario_id)
    assert set(validation.scenario_id).isdisjoint(test.scenario_id)
