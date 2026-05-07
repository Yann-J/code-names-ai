from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from codenames_ai.agent.spymaster import AISpymaster
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.eval.golden import GoldenBoard, GoodClueOption
from codenames_ai.game.models import Board, Card, Color
from codenames_ai.vocab.models import Vocabulary, VocabConfig


def _norm(v: list[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return (arr / n).astype(np.float32) if n > 0 else arr


def _fw(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _board(
    friendly: list[str],
    opponent: list[str],
    neutral: list[str],
    assassin: str,
    *,
    first_team: Color = Color.RED,
) -> Board:
    other = first_team.opponent()
    cards = (
        [Card(word=w, lemma=w, color=first_team) for w in friendly]
        + [Card(word=w, lemma=w, color=other) for w in opponent]
        + [Card(word=w, lemma=w, color=Color.NEUTRAL) for w in neutral]
        + [Card(word=assassin, lemma=assassin, color=Color.ASSASSIN)]
    )
    return Board(cards=tuple(cards), first_team=first_team)


def _setup(
    entries: list[tuple[str, list[float], float]],
) -> tuple[EmbeddingMatrix, Vocabulary]:
    surfaces = [e[0] for e in entries]
    vectors = np.stack([_norm(e[1]) for e in entries])
    matrix = EmbeddingMatrix(
        vectors=vectors,
        surfaces=surfaces,
        surface_to_index={s: i for i, s in enumerate(surfaces)},
        provider_id="golden-isolated",
        vocab_cache_key="golden-isolated",
    )
    config = VocabConfig(
        language="en",
        zipf_min=3.0,
        zipf_max=7.0,
        allowed_pos=frozenset({"NOUN", "ADJ"}),
    )
    df = pd.DataFrame(
        [{"surface": s, "lemma": s, "zipf": z, "pos": "NOUN"} for (s, _, z) in entries]
    )
    return matrix, Vocabulary(config=config, df=df)


def _golden_cluster_single_axis() -> tuple[GoldenBoard, AISpymaster]:
    z = 5.0
    af, ao, an = _fw("af", 9), _fw("ao", 8), _fw("an", 7)
    entries: list[tuple[str, list[float], float]] = []
    for i, w in enumerate(af):
        entries.append((w, [1.0, 0.01 * i], z))
    for i, w in enumerate(ao):
        entries.append((w, [-1.0, 0.01 * i], z))
    for i, w in enumerate(an):
        entries.append((w, [-0.5 + 0.02 * i, 0.5], z))
    entries.append(("aass", [-0.8, -0.3], z))
    entries.append(("aclue_good", [1.0, 0.0], z))
    entries.append(("aclue_bad", [-1.0, 0.0], z))
    m, v = _setup(entries)
    board = _board(af, ao, an, "aass")
    golden = GoldenBoard(
        name="cluster_single_axis",
        board=board,
        team=Color.RED,
        options=(GoodClueOption(frozenset(af), "all_friendlies"),),
    )
    return golden, AISpymaster(m, v, risk=0.5, top_k=20)


def _golden_two_bands() -> tuple[GoldenBoard, AISpymaster]:
    z = 5.0
    bf, bo, bn = _fw("bf", 9), _fw("bo", 8), _fw("bn", 7)
    entries: list[tuple[str, list[float], float]] = []
    for i in range(5):
        entries.append((bf[i], [1.0, 0.01 * i], z))
    for i in range(5, 9):
        entries.append((bf[i], [0.0, 1.0 + 0.01 * (i - 5)], z))
    for i, w in enumerate(bo):
        entries.append((w, [-1.0, 0.02 * i], z))
    for i, w in enumerate(bn):
        entries.append((w, [0.4, 0.5 + 0.02 * i], z))
    entries.append(("bass", [-0.7, -0.5], z))
    entries.append(("bclue_a", [1.0, 0.0], z))
    entries.append(("bclue_b", [0.0, 1.0], z))
    m, v = _setup(entries)
    board = _board(bf, bo, bn, "bass")
    golden = GoldenBoard(
        name="two_orthogonal_clusters",
        board=board,
        team=Color.RED,
        options=(
            GoodClueOption(frozenset(bf[:5]), "band_a"),
            GoodClueOption(frozenset(bf[5:]), "band_b"),
        ),
    )
    return golden, AISpymaster(m, v, risk=0.5, top_k=20)


def _golden_assassin_gap() -> tuple[GoldenBoard, AISpymaster]:
    z = 5.0
    cf, co, cn = _fw("cf", 9), _fw("co", 8), _fw("cn", 7)
    entries: list[tuple[str, list[float], float]] = []
    for i, w in enumerate(cf):
        entries.append((w, [1.0, 0.01 * i], z))
    for i, w in enumerate(co):
        entries.append((w, [-0.2, -1.0 + 0.02 * i], z))
    for i, w in enumerate(cn):
        entries.append((w, [-0.4 + 0.02 * i, 0.3], z))
    entries.append(("cass", [-0.95, -0.35], z))
    entries.append(("cclue_safe", [1.0, 0.02], z))
    entries.append(("cclue_risky", [-0.95, -0.3], z))
    m, v = _setup(entries)
    board = _board(cf, co, cn, "cass")
    golden = GoldenBoard(
        name="avoid_assassin_direction",
        board=board,
        team=Color.RED,
        options=(GoodClueOption(frozenset(cf), "friendly_plane"),),
    )
    return golden, AISpymaster(m, v, risk=0.5, top_k=20)


def iter_golden_cases(
    *,
    risk: float = 0.5,
    top_k: int = 20,
) -> Iterator[tuple[GoldenBoard, AISpymaster]]:
    """Yield `(golden, spymaster)` pairs in isolated synthetic spaces."""
    for g, spy in (
        _golden_cluster_single_axis(),
        _golden_two_bands(),
        _golden_assassin_gap(),
    ):
        if risk != 0.5 or top_k != 20:
            spy = AISpymaster(spy.matrix, spy.clue_vocabulary, risk=risk, top_k=top_k)
        yield g, spy


def default_golden_boards() -> tuple[tuple[GoldenBoard, ...], AISpymaster]:
    """Return `(goldens, example_spymaster)` for quick checks; eval uses `iter_golden_cases`."""
    pairs = list(iter_golden_cases())
    goldens = tuple(g for g, _ in pairs)
    return goldens, pairs[0][1]
