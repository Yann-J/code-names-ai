from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from codenames_ai.cli.eval_config import load_eval_yaml
from codenames_ai.cli.runtime import build_eval_runtime
from codenames_ai.config import Config
from codenames_ai.embedding.download import download_fasttext
from codenames_ai.eval.metrics import aggregate, compare
from codenames_ai.eval.persist import save_records
from codenames_ai.eval.tournament import run_tournament
from codenames_ai.learn import LeagueConfig, run_league_learning
from codenames_ai.storage import StoragePaths


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s %(message)s",
    )


def cmd_download_fasttext(args: argparse.Namespace) -> int:
    app = Config()
    storage = StoragePaths.from_config(app)
    storage.ensure()
    path = download_fasttext(args.lang, storage.models_dir, force=args.force)
    print(path)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    app = Config()
    storage = StoragePaths.from_config(app)
    storage.ensure()

    paths = [Path(p).resolve() for p in args.config]
    loaded = [load_eval_yaml(p) for p in paths]

    label_to_records: dict[str, list] = {}
    all_records = []

    for path, (yaml_cfg, digest) in zip(paths, loaded, strict=True):
        cfg = yaml_cfg.model_copy(
            update={
                "scoring": yaml_cfg.scoring.model_copy(
                    update={
                        "llm_rerank": yaml_cfg.scoring.llm_rerank
                        and not args.embedding_only
                    }
                )
            }
        )
        label = cfg.label
        rt = build_eval_runtime(cfg, app)
        if len(rt.game_vocab) < 25:
            print(
                f"error: game vocabulary has only {len(rt.game_vocab)} words "
                f"(need ≥ 25). Check Zipf/POS filters.",
                file=sys.stderr,
            )
            return 1

        seeds = list(range(args.seed_start, args.seed_start + args.runs))
        records = run_tournament(
            seeds=seeds,
            game_vocab=rt.game_vocab,
            red_spymaster=rt.spymaster,
            red_guesser=rt.guesser,
            blue_spymaster=rt.spymaster,
            blue_guesser=rt.guesser,
            max_clues=args.max_clues,
            label=label,
            config_hash=digest,
        )
        label_to_records.setdefault(label, []).extend(records)
        all_records.extend(records)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")
    out = Path(args.output) if args.output else storage.evals_dir / f"{run_id}.parquet"
    save_records(all_records, out)
    print(f"wrote {out} ({len(all_records)} rows)")

    if len(loaded) > 1:
        rows = compare(label_to_records)
        print("\ncomparison (aggregate per config label):")
        for row in sorted(rows, key=lambda r: r["label"]):
            print(
                f"  {row['label']!r}: n={int(row['n_games'])} "
                f"red_win={row['red_win_rate']:.2f} blue_win={row['blue_win_rate']:.2f} "
                f"assassin={row['assassin_rate']:.2f} "
                f"avg_clues={row['avg_clues']:.1f} "
                f"avg_g/clue={row['avg_guesses_per_clue']:.2f}"
            )
    else:
        ag = aggregate(all_records)
        print(
            "\nsummary:",
            f"n={int(ag['n_games'])} red_win={ag['red_win_rate']:.2f} "
            f"assassin={ag['assassin_rate']:.2f} avg_clues={ag['avg_clues']:.1f}",
        )

    return 0


def cmd_golden(args: argparse.Namespace) -> int:
    from codenames_ai.eval.golden import evaluate_golden, golden_pass_rate
    from codenames_ai.eval.golden_boards import iter_golden_cases

    results = [
        evaluate_golden(spy, g)
        for g, spy in iter_golden_cases(risk=args.risk, top_k=args.top_k)
    ]
    for r in results:
        mark = "ok" if r.matched else "FAIL"
        print(
            f"[{mark}] {r.name}: clue={r.chosen_clue!r} n={r.chosen_n} "
            f"targets={r.chosen_targets}"
        )
    pr = golden_pass_rate(results)
    print(f"pass rate: {pr:.0%}")
    return 0 if pr == 1.0 else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from codenames_ai.web.app import create_app

    agent_cfg = None
    if args.config:
        cfg_path = Path(args.config).resolve()
        agent_cfg, _ = load_eval_yaml(cfg_path)

    app = create_app(agent_config=agent_cfg)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.verbose else "info",
    )
    return 0


def cmd_learn_league(args: argparse.Namespace) -> int:
    app = Config()
    cfg_path = Path(args.config).resolve()
    eval_cfg, digest = load_eval_yaml(cfg_path)
    league_cfg = LeagueConfig(
        generations=args.generations,
        jobs=args.jobs,
        verbose=args.verbose,
    )
    root = Path(args.output_dir).resolve()
    run_dir = run_league_learning(
        app_cfg=app,
        eval_cfg=eval_cfg,
        eval_config_hash=digest,
        league_cfg=league_cfg,
        seed=args.seed,
        root_dir=root,
        run_id=args.run_id,
        resume=args.resume,
        cli_args={
            "config": str(cfg_path),
            "generations": args.generations,
            "seed": args.seed,
            "jobs": args.jobs,
            "output_dir": str(root),
            "run_id": args.run_id,
            "resume": args.resume,
        },
    )
    print(f"done: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codenames-ai",
        description="Code Names AI — embeddings, agents, eval, and API/PWA serve.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="Fetch optional model weights")
    d_sub = d.add_subparsers(dest="what", required=True)
    ft = d_sub.add_parser("fasttext", help="Download fastText cc.*.300.bin (large)")
    ft.add_argument("--lang", default="en", help="Language code (default: en)")
    ft.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the .bin already exists",
    )
    ft.set_defaults(func=cmd_download_fasttext)

    ev = sub.add_parser("eval", help="Self-play tournament → parquet + summary table")
    ev.add_argument("--runs", type=int, required=True, help="Number of games (seeds)")
    ev.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First RNG seed (default: 0)",
    )
    ev.add_argument(
        "--config",
        action="append",
        required=True,
        dest="config",
        metavar="PATH",
        help="Agent YAML (repeat for side-by-side comparison)",
    )
    ev.add_argument(
        "--embedding-only",
        action="store_true",
        help="Ignore scoring.llm_rerank in YAML and run embedding-only agents",
    )
    ev.add_argument(
        "--max-clues",
        type=int,
        default=50,
        help="Cap clues per game before declaring no winner",
    )
    ev.add_argument(
        "--output",
        default=None,
        help="Parquet path (default: ~/.cache/codenames_ai/evals/<run_id>.parquet)",
    )
    ev.add_argument(
        "--run-id",
        default=None,
        help="Filename stem when --output is omitted (default: timestamp)",
    )
    ev.set_defaults(func=cmd_eval)

    go = sub.add_parser("golden", help="Run curated golden-board spymaster checks (synthetic embeddings)")
    go.add_argument("--risk", type=float, default=0.5)
    go.add_argument("--top-k", type=int, default=20, dest="top_k")
    go.set_defaults(func=cmd_golden)

    sv = sub.add_parser("serve", help="FastAPI API + React PWA shell")
    sv.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Agent YAML (same schema as eval --config; default: web embedding-only preset)",
    )
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    ll = sub.add_parser("learn-league", help="Tune spymaster scoring via league self-play")
    ll.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Base eval YAML used for vocab/runtime and fixed non-learned settings",
    )
    ll.add_argument("--generations", type=int, default=30)
    ll.add_argument("--seed", type=int, default=0, help="Global RNG seed")
    ll.add_argument(
        "--jobs",
        type=int,
        default=max(1, ((os.cpu_count() or 2) - 1)),
        help="Parallel worker processes (default: cpu_count-1)",
    )
    ll.add_argument(
        "--output-dir",
        default="./league",
        help="Artifact root directory (default: ./league)",
    )
    ll.add_argument("--run-id", default=None, help="Explicit run directory name under output-dir")
    ll.add_argument(
        "--resume",
        default=None,
        metavar="RUN_ID",
        help="Resume from output-dir/RUN_ID checkpoint_latest.json",
    )
    ll.set_defaults(func=cmd_learn_league)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
