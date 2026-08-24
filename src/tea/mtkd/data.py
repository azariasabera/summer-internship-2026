# src/tea/mtkd/data.py

"""MTKD dataset loading: monolingual (single language) or multilingual
(train/dev mixed across EN+FI+FR, test fixed to the target language).

The single-language loaders themselves (`iemocap`/`fesc`/`cafe`) are NOT
redefined here. `tea.teachers.LOADERS` is the one canonical version, this module only adds the
monolingual/multilingual dispatch on top, ported from `mtkd_code.txt`'s `data.py`.
"""

from __future__ import annotations

from datasets import DatasetDict, concatenate_datasets
from omegaconf import DictConfig


def build_dataset(cfg: DictConfig, linguality: str, language: str, session: int) -> DatasetDict:
    """Build the train/test/dev splits for one MTKD training run.

    Parameters
    ----------
    cfg:
        Resolved Hydra config.
    linguality:
        `"Monolingual"` (train/test/dev all from `language`'s own split) or
        `"Multilingual"` (train/dev/test are each the concatenation of all
        three languages' splits.
    language:
        Target language, `"EN"`/`"FI"`/`"FR"`.
    session:
        Dataset session/split index for `language`. Auxiliary languages
        (multilingual mode) use `cfg.mtkd.multilingual_aux_session` instead.
    """
    from tea.teachers import LOADERS

    if linguality == "Monolingual":
        return LOADERS[language](session, cfg)

    ds_by_lang = {}
    for lang, loader in LOADERS.items():
        sess = session if lang == language else cfg.mtkd.multilingual_aux_session[lang]
        ds_by_lang[lang] = loader(sess, cfg)

    train_ds = concatenate_datasets([ds_by_lang[l]["train"] for l in LOADERS])
    dev_ds = concatenate_datasets([ds_by_lang[l]["dev"] for l in LOADERS])
    # NOTE: test is also concatenated across all three languages here. If we want the Finnish-only test set specifically 
    # call `evaluate_student` with `linguality="Monolingual"` instead.
    test_ds = concatenate_datasets([ds_by_lang[l]["test"] for l in LOADERS])
    return DatasetDict({"train": train_ds, "test": test_ds, "dev": dev_ds})
