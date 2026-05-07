from __future__ import annotations

import logging

import pandas as pd

from codenames_ai.storage import StoragePaths
from codenames_ai.vocab.filters import is_valid_surface, load_exclusions
from codenames_ai.vocab.frequency import FrequencyProvider, WordfreqProvider
from codenames_ai.vocab.linguistic import LinguisticProcessor, SpacyLinguisticProcessor
from codenames_ai.vocab.models import Vocabulary, VocabConfig

logger = logging.getLogger(__name__)


def build_vocabulary(
    config: VocabConfig,
    *,
    frequency_provider: FrequencyProvider | None = None,
    linguistic: LinguisticProcessor | None = None,
) -> Vocabulary:
    """Build a `Vocabulary` from sources without touching disk.

    Stages:
      1. Pull surface forms from the frequency provider within the Zipf window.
      2. Apply char-set / length / exclusion filters.
      3. POS-tag and lemmatize survivors via the linguistic processor.
      4. POS-filter to `config.allowed_pos` and assemble the DataFrame.
    """
    if frequency_provider is None:
        frequency_provider = WordfreqProvider()
    if linguistic is None:
        linguistic = SpacyLinguisticProcessor.for_language(config.language)

    exclusions = load_exclusions(config.exclusions_path)

    candidates: list[tuple[str, float]] = []
    for surface, zipf in frequency_provider.iter_range(
        language=config.language,
        zipf_min=config.zipf_min,
        zipf_max=config.zipf_max,
    ):
        surface_lower = surface.lower()
        if surface_lower in exclusions:
            continue
        if not is_valid_surface(
            surface_lower,
            min_length=config.min_length,
            allow_hyphens=config.allow_hyphens,
        ):
            continue
        candidates.append((surface_lower, zipf))

    logger.info("vocab pre-linguistic candidates: %d", len(candidates))

    if not candidates:
        return Vocabulary(
            config=config,
            df=pd.DataFrame(columns=["surface", "lemma", "zipf", "pos"]),
        )

    surfaces = [s for s, _ in candidates]
    analyses = linguistic.analyze_batch(surfaces)

    rows: list[dict] = []
    for (surface, zipf), (lemma, pos) in zip(candidates, analyses, strict=True):
        if pos not in config.allowed_pos:
            continue
        rows.append({"surface": surface, "lemma": lemma, "zipf": zipf, "pos": pos})

    df = pd.DataFrame(rows, columns=["surface", "lemma", "zipf", "pos"])
    logger.info("vocab final entries: %d", len(df))
    return Vocabulary(config=config, df=df)


def load_or_build_vocabulary(
    config: VocabConfig,
    storage: StoragePaths,
    *,
    frequency_provider: FrequencyProvider | None = None,
    linguistic: LinguisticProcessor | None = None,
) -> Vocabulary:
    """Load a cached vocabulary if available; otherwise build, save, and return one."""
    cache_path = storage.vocab_dir_for(config.language) / f"{config.cache_key()}.parquet"
    if cache_path.exists():
        logger.info("vocab cache hit: %s", cache_path)
        return Vocabulary.load(config, cache_path)

    logger.info("vocab cache miss; building (key=%s)", config.cache_key())
    vocab = build_vocabulary(
        config,
        frequency_provider=frequency_provider,
        linguistic=linguistic,
    )
    vocab.save(cache_path)
    return vocab
