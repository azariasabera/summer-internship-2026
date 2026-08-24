# src/tea/teachers/data.py

"""Dataset loaders for monolingual teacher training.

Provides canonical IEMOCAP, FESC, and CaFE dataset loaders and
returns Hugging Face DatasetDict objects with train/dev/test splits.
"""

from __future__ import annotations

import json

import pandas as pd
from datasets import Audio, Dataset, DatasetDict
from omegaconf import DictConfig

from tea.utils.constants import CLASS_ORDER, LABEL2ID
from tea.utils.paths import resolve

def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(df[["audio", "label"]])
    return ds.cast_column("audio", Audio(sampling_rate=16_000))


def _update_iemocap_path(path: str, datasets_root: str, iemocap_old_root: str) -> str:
    return datasets_root + path.split(iemocap_old_root)[-1]


def _update_iemocap_label(label: str) -> str:
    return {"ang": "anger", "hap": "happiness", "neu": "neutral", "sad": "sadness"}[label]


def _update_cafe_path(path: str, cafe_root: str, cafe_old_root: str) -> str:
    return cafe_root + path.split(cafe_old_root)[-1]


def _update_fesc_label(label: str) -> str:
    return {"1": "neutral", "2": "sadness", "3": "happiness", "4": "anger"}[label]


def _is_common(emotion: str) -> int:
    return int(emotion in CLASS_ORDER)


def iemocap(session: int, cfg: DictConfig) -> DatasetDict:
    """Load one IEMOCAP session's train/test splits (dev = test, matching the original setup).

    Parameters
    ----------
    session:
        Session number (1-5).
    cfg:
        Resolved Hydra config (reads `cfg.paths.splits_root`/`datasets_root`
        and `cfg.teachers.iemocap_old_root`).
    """
    splits_root = resolve(cfg.paths.splits_root)
    with open(splits_root / "iemocap" / f"session{session}" / "train.json") as f:
        train_data = json.load(f)
    with open(splits_root / "iemocap" / f"session{session}" / "test.json") as f:
        test_data = json.load(f)

    def build(data: dict) -> Dataset:
        df = pd.DataFrame.from_dict(data, orient="index").reset_index()
        df = df.rename(columns={"index": "file_id", "wav": "audio"})
        df["audio"] = df["audio"].apply(
            lambda p: _update_iemocap_path(p, str(resolve(cfg.paths.datasets_root)), cfg.teachers.iemocap_old_root)
        )
        df["emo"] = df["emo"].apply(_update_iemocap_label)
        df["label"] = df["emo"].map(LABEL2ID)
        return _to_hf_dataset(df)

    test_ds = build(test_data)
    return DatasetDict({"train": build(train_data), "test": test_ds, "dev": test_ds})


def fesc(session: int, cfg: DictConfig) -> DatasetDict:
    """Load one FESC (Finnish) speaker session's train/test/dev splits.

    Parameters
    ----------
    session:
        Session number (1-9), mapped to a speaker folder via `cfg.teachers.fesc_session_map`.
    cfg:
        Resolved Hydra config.
    """
    folder = cfg.teachers.fesc_session_map[session]
    splits_root = resolve(cfg.paths.splits_root)
    fesc_new_prefix = str(resolve(cfg.paths.datasets_root) / "FESC") + "/"

    splits = {}
    for split in ("train", "test", "dev"):
        with open(splits_root / "Finnish-emotion-spilits" / folder / f"{split}.json") as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient="index").reset_index()
        df["audio"] = df["file_path"].str.replace(cfg.teachers.fesc_old_prefix, fesc_new_prefix, regex=False)
        df = df.rename(columns={"index": "file_id", "label": "emo"})
        df = df.loc[df["emo"] != "5"].reset_index(drop=True)  # drop the 5th (non-canonical) class
        df["emo"] = df["emo"].apply(_update_fesc_label)
        df["label"] = df["emo"].map(LABEL2ID)
        splits[split] = _to_hf_dataset(df)

    return DatasetDict(splits)


def cafe(session: int | None, cfg: DictConfig) -> DatasetDict:
    """Load CaFE (French) train/test/dev splits, keeping only the 4 canonical emotion classes.

    Parameters
    ----------
    session:
        Unused (CaFE has one split set) -- kept for a uniform `LOADERS[lang](session, cfg)` signature.
    cfg:
        Resolved Hydra config.
    """
    splits_root = resolve(cfg.paths.splits_root)
    cafe_root = str(resolve(cfg.paths.cafe_root))

    frames = []
    for file in sorted((splits_root / "CaFE_json_splits").glob("*")):
        with open(file) as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient="index").reset_index()
        df["set"] = file.stem
        df = df.rename(columns={"wav": "audio", "emo": "label"})
        df["label_flag"] = df["label"].apply(_is_common)
        frames.append(df)

    cafe_df = pd.concat(frames, axis=0)
    cafe_df = cafe_df.loc[cafe_df["label_flag"] == 1].reset_index(drop=True)
    cafe_df["audio"] = cafe_df["audio"].apply(lambda p: _update_cafe_path(p, cafe_root, cfg.teachers.cafe_old_root))
    cafe_df["label"] = cafe_df["label"].map(LABEL2ID)

    train_df = cafe_df.loc[cafe_df["set"] != "test", ["audio", "label"]].reset_index(drop=True)
    test_df = cafe_df.loc[cafe_df["set"] == "test", ["audio", "label"]].reset_index(drop=True)

    test_ds = _to_hf_dataset(test_df)
    return DatasetDict({"train": _to_hf_dataset(train_df), "test": test_ds, "dev": test_ds})


LOADERS = {"EN": iemocap, "FI": fesc, "FR": cafe}