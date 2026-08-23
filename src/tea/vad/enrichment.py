# src/tea/vad/enrichment.py

"""Utilities for enriching fixed-size segments with VAD annotations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from typing import Union

def enrich_fixed_segments_from_vad(fixed_segments: dict, vad_csv: Union[str, Path], output_json: Union[str, Path], sr: int = 16_000) -> None:
    """Inherit ground-truth labels/confidence/child-speech from VAD-based annotations onto fixed-size segments.

    For each fixed-size chunk, finds every overlapping VAD-annotated
    segment, then assigns the label with the highest
    contribution-weighted-by-confidence score (contribution = fraction of
    the fixed chunk covered by that VAD segment). Child-speech presence is
    contribution-weighted-averaged independently of the label.

    Parameters
    ----------
    fixed_segments:
        One entry from `Segmenter.chunk_fixed`'s output.
    vad_csv:
        Annotation CSV (VAD-based, already labeled) to inherit from.
    output_json:
        Where to write the enriched segment metadata.
    sr:
        Sample rate used to convert chunk duration to seconds.
    """
    vad = pd.read_csv(vad_csv)
    enriched_segments = []

    for chunk in fixed_segments["segments"]:
        c_start, c_end = chunk["start"], chunk["end"]
        chunk_duration = (c_end - c_start) / sr

        overlaps = []
        for _, row in vad.iterrows():
            v_start, v_end = int(row["start"]), int(row["end"])
            overlap_samples = min(c_end, v_end) - max(c_start, v_start)
            if overlap_samples <= 0:
                continue

            contribution = overlap_samples / (c_end - c_start)
            overlaps.append(
                {
                    "type": row["type"],
                    "gt_label": row["gt_label"],
                    "confidence": row["confidence"],
                    "child_speech": row["overlap"],
                    "contribution": float(contribution),
                    "duration": float(contribution * chunk_duration),
                    "start": v_start,
                    "end": v_end,
                }
            )

        new_chunk = chunk.copy()
        new_chunk["overlapping_vad_segments"] = overlaps

        if not overlaps:
            new_chunk.update(
                {"inherited_gt_label": None, "inherited_confidence": None, "inherited_child_speech": None}
            )
            enriched_segments.append(new_chunk)
            continue

        # ---- label selection: contribution * confidence weighted vote ----
        label_candidates = [x for x in overlaps if pd.notna(x["gt_label"])]
        if label_candidates:
            label_scores: dict = {}
            for x in label_candidates:
                if pd.isna(x["confidence"]):
                    continue
                score = x["contribution"] * float(x["confidence"])
                label_scores[x["gt_label"]] = label_scores.get(x["gt_label"], 0) + score

            if label_scores:
                selected_label = max(label_scores, key=label_scores.get)
                new_chunk["inherited_gt_label"] = selected_label

                selected = [
                    x for x in label_candidates if x["gt_label"] == selected_label and pd.notna(x["confidence"])
                ]
                denom = sum(x["contribution"] for x in selected)
                new_chunk["inherited_confidence"] = (
                    sum(x["contribution"] * float(x["confidence"]) for x in selected) / denom
                    if denom > 0
                    else None
                )
            else:
                new_chunk["inherited_gt_label"] = None
                new_chunk["inherited_confidence"] = None
        else:
            new_chunk["inherited_gt_label"] = None
            new_chunk["inherited_confidence"] = None

        # ---- child speech: contribution-weighted average, independent of label ----
        child_candidates = [x for x in overlaps if pd.notna(x["child_speech"])]
        new_chunk["inherited_child_speech"] = (
            sum(x["contribution"] * float(x["child_speech"]) for x in child_candidates)
            if child_candidates
            else None
        )

        enriched_segments.append(new_chunk)

    output = fixed_segments.copy()
    output["segments"] = enriched_segments
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)