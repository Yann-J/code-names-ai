from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from codenames_ai.cli.eval_config import EvalAgentConfigFile
from codenames_ai.learn.league import (
    PARAM_SPECS,
    fitness_from_metrics,
    initial_sigmas,
    initialize_population,
    load_latest_checkpoint,
    mutate_params,
    rank_policies,
    save_checkpoint,
    validate_param_specs_within_schema_bounds,
)


def test_param_specs_within_eval_schema_bounds() -> None:
    validate_param_specs_within_schema_bounds()


def test_population_init_contains_baseline_and_is_seeded() -> None:
    cfg = EvalAgentConfigFile()
    sigmas = initial_sigmas()
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    pop_a = initialize_population(
        eval_cfg=cfg,
        pop_size=12,
        rng=rng_a,
        sigmas=sigmas,
    )
    pop_b = initialize_population(
        eval_cfg=cfg,
        pop_size=12,
        rng=rng_b,
        sigmas=sigmas,
    )
    assert pop_a[0].policy_id == "p000"
    assert pop_a[0].params == pop_b[0].params
    assert [p.params for p in pop_a] == [p.params for p in pop_b]


def test_mutation_respects_bounds() -> None:
    parent = {
        name: (low + high) / 2.0
        for name, (low, high) in PARAM_SPECS.items()
    }
    rng = np.random.default_rng(0)
    sigmas = {
        name: (high - low) * 10.0
        for name, (low, high) in PARAM_SPECS.items()
    }
    child = mutate_params(parent, sigmas=sigmas, rng=rng)
    for name, value in child.items():
        low, high = PARAM_SPECS[name]
        assert low <= value <= high


def test_fitness_and_tie_break_are_deterministic() -> None:
    metrics = [
        {
            "policy_id": "p001",
            "fitness": fitness_from_metrics(
                win_rate=0.5,
                avg_correct_guesses=2.0,
                assassin_rate=0.1,
            ),
            "win_rate": 0.5,
            "assassin_rate": 0.1,
            "params": {},
        },
        {
            "policy_id": "p002",
            "fitness": fitness_from_metrics(
                win_rate=0.5,
                avg_correct_guesses=2.0,
                assassin_rate=0.1,
            ),
            "win_rate": 0.5,
            "assassin_rate": 0.1,
            "params": {},
        },
    ]
    h2h = {("p001", "p002"): {"p001": 3, "p002": 1}}
    ranked = rank_policies(metrics, h2h)
    assert ranked[0]["policy_id"] == "p001"


def test_checkpoint_roundtrip_and_resume_loads_latest_generation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "league" / "run-test"
    pop = []
    rng = np.random.default_rng(7)
    save_checkpoint(
        run_dir=run_dir,
        generation=2,
        population=pop,
        metrics=[],
        sigmas=initial_sigmas(),
        rng=rng,
    )
    loaded = load_latest_checkpoint(run_dir)
    assert loaded["generation"] == 2
    assert "rng_state" in loaded


def test_checkpoint_writes_expected_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "league" / "run-test"
    rng = np.random.default_rng(7)
    save_checkpoint(
        run_dir=run_dir,
        generation=0,
        population=[],
        metrics=[],
        sigmas=initial_sigmas(),
        rng=rng,
    )
    assert (run_dir / "checkpoint_gen_000.json").exists()
    assert (run_dir / "checkpoint_latest.json").exists()
    payload = json.loads(
        (run_dir / "checkpoint_latest.json").read_text(encoding="utf-8")
    )
    assert payload["generation"] == 0
