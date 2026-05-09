from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from codenames_ai.agent.guesser import AIGuesser
from codenames_ai.agent.scoring import ScoringWeights
from codenames_ai.agent.spymaster import AISpymaster
from codenames_ai.cli.eval_config import EvalAgentConfigFile, ScoringConfig
from codenames_ai.cli.runtime import EvalRuntime, build_eval_runtime
from codenames_ai.config import Config
from codenames_ai.game.board import generate_board
from codenames_ai.game.models import Color
from codenames_ai.game.orchestrator import Game

PARAM_SPECS: dict[str, tuple[float, float]] = {
    "ambition_weight": (0.0, 2.0),
    "margin_weight": (0.0, 2.0),
    "assassin_weight": (0.0, 5.0),
    "opponent_weight": (0.0, 5.0),
    "undercluster_penalty_weight": (0.0, 2.0),
    "expected_reward_weight": (0.0, 5.0),
    "mc_temperature": (0.01, 5.0),
}

FITNESS_EPS = 1e-12

_WORKER_RUNTIME: EvalRuntime | None = None
_WORKER_CFG: EvalAgentConfigFile | None = None


@dataclass(frozen=True)
class LeagueConfig:
    generations: int = 30
    population_size: int = 12
    elites: int = 4
    random_injections: int = 2
    games_per_pair: int = 6
    seeds_refresh_every: int = 5
    plateau_patience: int = 8
    plateau_min_delta: float = 0.01
    max_clues: int = 50
    jobs: int = max(1, (os.cpu_count() or 2) - 1)
    verbose: bool = False


@dataclass(frozen=True)
class Policy:
    policy_id: str
    params: dict[str, float]


def _param_ranges() -> dict[str, float]:
    return {name: upper - lower for name, (lower, upper) in PARAM_SPECS.items()}


def validate_param_specs_within_schema_bounds() -> None:
    schema = ScoringConfig.model_json_schema()
    props = schema.get("properties", {})
    for name, (lower, upper) in PARAM_SPECS.items():
        info = props.get(name, {})
        minimum = info.get("minimum")
        maximum = info.get("maximum")
        if minimum is not None and lower < float(minimum):
            raise ValueError(f"PARAM_SPECS[{name}] lower={lower} below schema minimum={minimum}")
        if maximum is not None and upper > float(maximum):
            raise ValueError(f"PARAM_SPECS[{name}] upper={upper} above schema maximum={maximum}")


def fitness_from_metrics(*, win_rate: float, avg_correct_guesses: float, assassin_rate: float) -> float:
    return win_rate + 0.2 * avg_correct_guesses - 0.5 * assassin_rate


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _make_manifest(
    *,
    run_dir: Path,
    seed: int,
    app_cfg: Config,
    eval_cfg: EvalAgentConfigFile,
    eval_config_hash: str,
    cli_args: dict[str, Any],
) -> None:
    manifest = {
        "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "cache_dir": str(app_cfg.cache_dir),
        "eval_config_label": eval_cfg.label,
        "eval_config_hash": eval_config_hash,
        "param_specs": PARAM_SPECS,
        "cli_args": cli_args,
        "git_commit": _git_commit_hash(),
    }
    _atomic_json_write(run_dir / "run_manifest.json", manifest)


def _git_commit_hash() -> str | None:
    head = Path(".git") / "HEAD"
    if not head.exists():
        return None
    try:
        content = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if content.startswith("ref: "):
        ref_path = Path(".git") / content.split(" ", 1)[1]
        if ref_path.exists():
            try:
                return ref_path.read_text(encoding="utf-8").strip()
            except OSError:
                return None
        return None
    return content or None


def _policy_from_eval_cfg(eval_cfg: EvalAgentConfigFile) -> dict[str, float]:
    base = ScoringWeights.from_risk(eval_cfg.risk)
    out: dict[str, float] = {}
    for name in PARAM_SPECS:
        value = getattr(eval_cfg.scoring, name)
        out[name] = float(value if value is not None else getattr(base, name))
    return out


def _random_policy(rng: np.random.Generator, idx: int) -> Policy:
    params = {
        name: float(rng.uniform(low=low, high=high))
        for name, (low, high) in PARAM_SPECS.items()
    }
    return Policy(policy_id=f"p{idx:03d}", params=params)


def _clip_params(params: dict[str, float]) -> dict[str, float]:
    clipped: dict[str, float] = {}
    for name, value in params.items():
        lo, hi = PARAM_SPECS[name]
        clipped[name] = float(min(hi, max(lo, value)))
    return clipped


def initialize_population(
    *,
    eval_cfg: EvalAgentConfigFile,
    pop_size: int,
    rng: np.random.Generator,
    sigmas: dict[str, float],
) -> list[Policy]:
    baseline = Policy(policy_id="p000", params=_policy_from_eval_cfg(eval_cfg))
    out = [baseline]
    for idx in range(1, min(4, pop_size)):
        out.append(
            Policy(
                policy_id=f"p{idx:03d}",
                params=mutate_params(baseline.params, sigmas=sigmas, rng=rng),
            )
        )
    while len(out) < pop_size:
        out.append(_random_policy(rng, len(out)))
    return out


def initial_sigmas() -> dict[str, float]:
    ranges = _param_ranges()
    return {name: 0.10 * width for name, width in ranges.items()}


def mutate_params(
    parent: dict[str, float],
    *,
    sigmas: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    out = {}
    for name, value in parent.items():
        noise = float(rng.normal(0.0, sigmas[name]))
        out[name] = value + noise
    return _clip_params(out)


def maybe_anneal_sigmas(
    *,
    sigmas: dict[str, float],
    stagnant_generations: int,
) -> dict[str, float]:
    if stagnant_generations == 0 or stagnant_generations % 5 != 0:
        return sigmas
    out: dict[str, float] = {}
    for name, sigma in sigmas.items():
        lo, hi = PARAM_SPECS[name]
        floor = 0.02 * (hi - lo)
        out[name] = max(floor, sigma * 0.8)
    return out


def _ensure_league_dirs(root: Path, run_id: str | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rid = run_id or _new_run_id()
    run_dir = root / rid
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _checkpoint_payload(
    *,
    generation: int,
    population: list[Policy],
    metrics: list[dict[str, Any]],
    sigmas: dict[str, float],
    rng_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generation": generation,
        "population": [{"policy_id": p.policy_id, "params": p.params} for p in population],
        "metrics": metrics,
        "sigmas": sigmas,
        "rng_state": rng_state,
    }


def save_checkpoint(
    *,
    run_dir: Path,
    generation: int,
    population: list[Policy],
    metrics: list[dict[str, Any]],
    sigmas: dict[str, float],
    rng: np.random.Generator,
) -> None:
    payload = _checkpoint_payload(
        generation=generation,
        population=population,
        metrics=metrics,
        sigmas=sigmas,
        rng_state=rng.bit_generator.state,
    )
    _atomic_json_write(run_dir / f"checkpoint_gen_{generation:03d}.json", payload)
    _atomic_json_write(run_dir / "checkpoint_latest.json", payload)


def load_latest_checkpoint(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "checkpoint_latest.json"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint found in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_scoring_weights_for_policy(eval_cfg: EvalAgentConfigFile, params: dict[str, float]) -> ScoringWeights:
    base = ScoringWeights.from_risk(eval_cfg.risk)
    updates: dict[str, Any] = {
        "prefer_min_targets": eval_cfg.scoring.prefer_min_targets,
        "mc_trials": eval_cfg.scoring.mc_trials,
        "adaptive_mc_base_trials": eval_cfg.scoring.adaptive_mc_base_trials,
        "adaptive_mc_extra_trials": eval_cfg.scoring.adaptive_mc_extra_trials,
        "adaptive_mc_ev_band": eval_cfg.scoring.adaptive_mc_ev_band,
        "lane_target_fractions": tuple(eval_cfg.scoring.lane_target_fractions),
        "lane_quality_delta_ev": eval_cfg.scoring.lane_quality_delta_ev,
        "lane_max_n": eval_cfg.scoring.lane_max_n,
    }
    updates.update(params)
    return ScoringWeights(**{**base.__dict__, **updates})


def _init_worker(cfg_data: dict[str, Any]) -> None:
    global _WORKER_RUNTIME, _WORKER_CFG
    _WORKER_CFG = EvalAgentConfigFile.model_validate(cfg_data)
    _WORKER_RUNTIME = build_eval_runtime(_WORKER_CFG, Config())


def _run_one_game(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_RUNTIME is None or _WORKER_CFG is None:
        raise RuntimeError("league worker not initialized")
    seed = int(task["seed"])
    red_params = task["red_params"]
    blue_params = task["blue_params"]
    red_policy = task["red_policy"]
    blue_policy = task["blue_policy"]

    spy_red = AISpymaster(
        _WORKER_RUNTIME.matrix,
        _WORKER_RUNTIME.clue_vocab,
        risk=_WORKER_CFG.risk,
        top_k=_WORKER_CFG.top_k_trace,
        reranker=None,
        weights=_build_scoring_weights_for_policy(_WORKER_CFG, red_params),
    )
    spy_blue = AISpymaster(
        _WORKER_RUNTIME.matrix,
        _WORKER_RUNTIME.clue_vocab,
        risk=_WORKER_CFG.risk,
        top_k=_WORKER_CFG.top_k_trace,
        reranker=None,
        weights=_build_scoring_weights_for_policy(_WORKER_CFG, blue_params),
    )
    guesser_red = AIGuesser(
        _WORKER_RUNTIME.matrix,
        risk=_WORKER_CFG.risk,
        reranker=None,
        sampling_temperature=_WORKER_CFG.guesser.sampling_temperature,
        sampling_top_k=_WORKER_CFG.guesser.sampling_top_k,
        rng=np.random.default_rng(seed * 17 + 3),
    )
    guesser_blue = AIGuesser(
        _WORKER_RUNTIME.matrix,
        risk=_WORKER_CFG.risk,
        reranker=None,
        sampling_temperature=_WORKER_CFG.guesser.sampling_temperature,
        sampling_top_k=_WORKER_CFG.guesser.sampling_top_k,
        rng=np.random.default_rng(seed * 17 + 7),
    )
    board = generate_board(_WORKER_RUNTIME.game_vocab, seed=seed)
    game = Game(
        board,
        red_spymaster=spy_red,
        red_guesser=guesser_red,
        blue_spymaster=spy_blue,
        blue_guesser=guesser_blue,
        seed=seed,
        max_clues=int(task["max_clues"]),
    )
    final = game.play()
    n_guesses = sum(1 for ev in final.turn_history if ev.kind == "GUESS")
    n_correct = sum(
        1
        for ev in final.turn_history
        if ev.kind == "GUESS" and ev.outcome_color is not None and ev.outcome_color == ev.team
    )
    assassin_hit = any(
        ev.kind == "GUESS" and ev.outcome_color == Color.ASSASSIN for ev in final.turn_history
    )
    winner_policy: str | None = None
    if final.winner == Color.RED:
        winner_policy = red_policy
    elif final.winner == Color.BLUE:
        winner_policy = blue_policy
    return {
        "seed": seed,
        "first_team": board.first_team.value,
        "red_policy": red_policy,
        "blue_policy": blue_policy,
        "winner_policy": winner_policy,
        "num_guesses": n_guesses,
        "correct_guesses": n_correct,
        "assassin_hit": assassin_hit,
    }


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def rank_policies(metrics: list[dict[str, Any]], head_to_head: dict[tuple[str, str], dict[str, int]]) -> list[dict[str, Any]]:
    by_id = {m["policy_id"]: m for m in metrics}
    ranked = list(metrics)
    ranked.sort(
        key=lambda m: (
            -float(m["fitness"]),
            -_head_to_head_score(m["policy_id"], by_id, head_to_head),
            float(m["assassin_rate"]),
            m["policy_id"],
        )
    )
    return ranked


def _head_to_head_score(
    pid: str,
    by_id: dict[str, dict[str, Any]],
    head_to_head: dict[tuple[str, str], dict[str, int]],
) -> float:
    tied = [other for other, m in by_id.items() if abs(m["fitness"] - by_id[pid]["fitness"]) <= FITNESS_EPS]
    if len(tied) <= 1:
        return 0.0
    wins = 0
    total = 0
    for other in tied:
        if other == pid:
            continue
        key = _pair_key(pid, other)
        row = head_to_head.get(key)
        if not row:
            continue
        wins += row.get(pid, 0)
        total += row.get(pid, 0) + row.get(other, 0)
    return (wins / total) if total else 0.0


def _policy_metrics(games: list[dict[str, Any]], policy_id: str) -> dict[str, float]:
    wins = 0
    played = 0
    assassin_hits = 0
    correct = 0.0
    for g in games:
        role = None
        if g["red_policy"] == policy_id:
            role = "red"
        elif g["blue_policy"] == policy_id:
            role = "blue"
        if role is None:
            continue
        played += 1
        if g["winner_policy"] == policy_id:
            wins += 1
        assassin_hits += int(g["assassin_hit"])
        correct += float(g["correct_guesses"])
    win_rate = (wins / played) if played else 0.0
    avg_correct = (correct / played) if played else 0.0
    assassin_rate = (assassin_hits / played) if played else 0.0
    return {
        "win_rate": win_rate,
        "avg_correct_guesses": avg_correct,
        "assassin_rate": assassin_rate,
        "fitness": fitness_from_metrics(
            win_rate=win_rate,
            avg_correct_guesses=avg_correct,
            assassin_rate=assassin_rate,
        ),
    }


def _seed_plan_for_pair(
    *,
    eval_runtime: EvalRuntime,
    generation: int,
    pair_index: int,
    games_per_pair: int,
) -> list[int]:
    wanted_i_first = games_per_pair // 2
    wanted_j_first = games_per_pair - wanted_i_first
    i_first: list[int] = []
    j_first: list[int] = []
    cursor = generation * 1_000_000 + pair_index * 1_000 + 11
    while len(i_first) < wanted_i_first or len(j_first) < wanted_j_first:
        board = generate_board(eval_runtime.game_vocab, seed=cursor)
        if board.first_team == Color.RED and len(i_first) < wanted_i_first:
            i_first.append(cursor)
        elif board.first_team == Color.BLUE and len(j_first) < wanted_j_first:
            j_first.append(cursor)
        cursor += 1
    return i_first + j_first


def _build_generation_tasks(
    *,
    eval_runtime: EvalRuntime,
    generation: int,
    population: list[Policy],
    league_cfg: LeagueConfig,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    pair_index = 0
    seed_epoch = generation // max(1, league_cfg.seeds_refresh_every)
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            pi = population[i]
            pj = population[j]
            seeds = _seed_plan_for_pair(
                eval_runtime=eval_runtime,
                generation=seed_epoch,
                pair_index=pair_index,
                games_per_pair=league_cfg.games_per_pair,
            )
            for seed in seeds:
                tasks.append(
                    {
                        "seed": seed,
                        "red_policy": pi.policy_id,
                        "blue_policy": pj.policy_id,
                        "red_params": pi.params,
                        "blue_params": pj.params,
                        "max_clues": league_cfg.max_clues,
                    }
                )
            pair_index += 1
    return tasks


def _evaluate_generation(
    *,
    eval_cfg: EvalAgentConfigFile,
    eval_runtime: EvalRuntime,
    generation: int,
    population: list[Policy],
    league_cfg: LeagueConfig,
    jobs: int,
    game_log_path: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, int]]]:
    cfg_data = eval_cfg.model_dump(mode="json")
    tasks = _build_generation_tasks(
        eval_runtime=eval_runtime,
        generation=generation,
        population=population,
        league_cfg=league_cfg,
    )
    games: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(cfg_data,)) as pool:
        futures = [pool.submit(_run_one_game, task) for task in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            games.append(result)
            with game_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"generation": generation, **result}) + "\n")
    h2h: dict[tuple[str, str], dict[str, int]] = {}
    for g in games:
        key = _pair_key(g["red_policy"], g["blue_policy"])
        row = h2h.setdefault(key, {g["red_policy"]: 0, g["blue_policy"]: 0})
        if g["winner_policy"] is not None:
            row[g["winner_policy"]] = row.get(g["winner_policy"], 0) + 1
    metrics: list[dict[str, Any]] = []
    for p in population:
        m = _policy_metrics(games, p.policy_id)
        metrics.append(
            {
                "policy_id": p.policy_id,
                "params": p.params,
                **m,
            }
        )
    return metrics, h2h


def _next_generation(
    *,
    ranked: list[dict[str, Any]],
    rng: np.random.Generator,
    league_cfg: LeagueConfig,
    sigmas: dict[str, float],
) -> list[Policy]:
    elites = [
        Policy(policy_id=ranked[idx]["policy_id"], params=dict(ranked[idx]["params"]))
        for idx in range(min(league_cfg.elites, len(ranked)))
    ]
    out = list(elites)
    target = league_cfg.population_size
    while len(out) < max(0, target - league_cfg.random_injections):
        parent = elites[int(rng.integers(0, len(elites)))]
        child = mutate_params(parent.params, sigmas=sigmas, rng=rng)
        out.append(Policy(policy_id=f"p{len(out):03d}", params=child))
    while len(out) < target:
        out.append(_random_policy(rng, len(out)))
    return out


def _print_generation(generation: int, ranked: list[dict[str, Any]], plateau: int, *, verbose: bool) -> None:
    best = ranked[0]
    print(
        f"gen={generation:03d} best={best['fitness']:.4f} "
        f"pid={best['policy_id']} plateau={plateau}"
    )
    for row in ranked[:3]:
        print(
            f"  {row['policy_id']}: fitness={row['fitness']:.4f} "
            f"win={row['win_rate']:.3f} assassin={row['assassin_rate']:.3f}"
        )
    if verbose:
        print("  sig: " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(best["params"].items())))


def run_league_learning(
    *,
    app_cfg: Config,
    eval_cfg: EvalAgentConfigFile,
    eval_config_hash: str,
    league_cfg: LeagueConfig,
    seed: int,
    root_dir: Path,
    run_id: str | None = None,
    resume: str | None = None,
    cli_args: dict[str, Any] | None = None,
) -> Path:
    validate_param_specs_within_schema_bounds()
    eval_cfg = eval_cfg.model_copy(update={"scoring": eval_cfg.scoring.model_copy(update={"llm_rerank": False})})
    rng = np.random.default_rng(seed)
    sigmas = initial_sigmas()

    if resume:
        run_dir = root_dir / resume
        ckpt = load_latest_checkpoint(run_dir)
        start_gen = int(ckpt["generation"]) + 1
        population = [Policy(policy_id=r["policy_id"], params=dict(r["params"])) for r in ckpt["population"]]
        sigmas = {k: float(v) for k, v in ckpt["sigmas"].items()}
        rng.bit_generator.state = ckpt["rng_state"]
    else:
        run_dir = _ensure_league_dirs(root_dir, run_id)
        start_gen = 0
        population = initialize_population(eval_cfg=eval_cfg, pop_size=league_cfg.population_size, rng=rng, sigmas=sigmas)
        _make_manifest(
            run_dir=run_dir,
            seed=seed,
            app_cfg=app_cfg,
            eval_cfg=eval_cfg,
            eval_config_hash=eval_config_hash,
            cli_args=cli_args or {},
        )

    print(f"league artifacts: {run_dir.resolve()}")
    history_rows: list[dict[str, Any]] = []
    game_log_path = run_dir / "games.jsonl"
    best_so_far = float("-inf")
    stagnant = 0
    hof: list[dict[str, Any]] = []
    eval_runtime = build_eval_runtime(eval_cfg, app_cfg)

    for generation in range(start_gen, league_cfg.generations):
        metrics, h2h = _evaluate_generation(
            eval_cfg=eval_cfg,
            eval_runtime=eval_runtime,
            generation=generation,
            population=population,
            league_cfg=league_cfg,
            jobs=max(1, league_cfg.jobs),
            game_log_path=game_log_path,
        )
        ranked = rank_policies(metrics, h2h)
        _print_generation(generation, ranked, stagnant, verbose=league_cfg.verbose)
        for m in ranked:
            history_rows.append({"generation": generation, **m})
        best_now = float(ranked[0]["fitness"])
        hof.extend(ranked[:2])
        hof = sorted(hof, key=lambda x: (-float(x["fitness"]), x["policy_id"]))[:5]

        improved = best_now > (best_so_far + league_cfg.plateau_min_delta)
        if improved:
            best_so_far = best_now
            stagnant = 0
        else:
            stagnant += 1
        sigmas = maybe_anneal_sigmas(sigmas=sigmas, stagnant_generations=stagnant)
        save_checkpoint(
            run_dir=run_dir,
            generation=generation,
            population=[Policy(policy_id=m["policy_id"], params=dict(m["params"])) for m in ranked],
            metrics=ranked,
            sigmas=sigmas,
            rng=rng,
        )
        if stagnant >= league_cfg.plateau_patience:
            print(f"early stop: plateau ({stagnant} generations)")
            break
        population = _next_generation(ranked=ranked, rng=rng, league_cfg=league_cfg, sigmas=sigmas)

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_parquet(run_dir / "history.parquet", index=False)
    best = max(hof, key=lambda x: x["fitness"])
    (run_dir / "best_policy.yaml").write_text(
        yaml.safe_dump({"policy_id": best["policy_id"], "params": best["params"]}, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "hall_of_fame.yaml").write_text(
        yaml.safe_dump(
            [{"policy_id": row["policy_id"], "fitness": row["fitness"], "params": row["params"]} for row in hof],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return run_dir
