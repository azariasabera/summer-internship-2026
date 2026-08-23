# src/tea/vad/refinement.py

"""VAD timeline construction and segment refinement."""

from __future__ import annotations

def build_full_timeline(speech_ts: list[dict], total_samples: int) -> list[dict]:
    """Turn Silero's speech-only timestamps into a full speech/non-speech timeline.

    Parameters
    ----------
    speech_ts:
        Silero `get_speech_timestamps` output: list of `{"start", "end"}` (samples).
    total_samples:
        Total length of the audio in samples.
    """
    speech_ts = sorted(speech_ts, key=lambda x: x["start"])

    segments = []
    prev_end = 0

    for s in speech_ts:
        if s["start"] > prev_end:
            segments.append({"start": prev_end, "end": s["start"], "type": "non-speech"})
        segments.append({"start": s["start"], "end": s["end"], "type": "speech"})
        prev_end = s["end"]

    if prev_end < total_samples:
        segments.append({"start": prev_end, "end": total_samples, "type": "non-speech"})

    return segments


def refine_segments(
    segments: list[dict],
    sr: int = 16_000,
    max_merge_gap: float = 2.0,
    min_speech_ratio: float = 0.7,
    min_speech_duration: float = 2.0,
    min_non_speech_duration: float = 2.0,
    max_segment_duration: float = 10.0,
    overlap: float = 1.0,
) -> list[dict]:
    """Post-process raw VAD segments into a cleaned, duration-bounded segment list.

    Pipeline (applied in order):
      1. Bridge short silences: a speech-silence-speech triplet is merged
         into one speech segment if the gap is short enough, the resulting
         speech-to-total ratio is high enough, and the merged duration
         doesn't exceed `max_segment_duration`. Repeats until no more merges
         are possible.
      2. Enforce min speech: any speech segment shorter than
         `min_speech_duration` is reclassified as non-speech.
      3. Merge consecutive non-speech segments into single spans.
      4. Remove short silences: non-speech spans shorter than
         `min_non_speech_duration` are dropped.
      5. Split long segments: anything longer than `max_segment_duration` is
         cut into bounded chunks. Speech chunks step back by `overlap`
         between cuts (to avoid cutting a word); non-speech chunks are cut
         back-to-back. A trailing piece shorter than the relevant minimum
         duration is absorbed into the preceding chunk instead of emitted
         on its own.

    Parameters
    ----------
    segments:
        Sorted, contiguous, non-overlapping `{"start", "end", "type"}` dicts
        (sample indices).
    sr, max_merge_gap, min_speech_ratio, min_speech_duration,
    min_non_speech_duration, max_segment_duration, overlap:
        See `conf/vad/vad.yaml` for the values used in the progress report.

    Returns
    -------
    list[dict]
        Refined segments in the same format. Empty list if input is empty.
    """
    if not segments:
        return []

    bridged = list(segments)
    speech_content = [
        seg["end"] - seg["start"] if seg["type"] == "speech" else 0 for seg in bridged
    ]
    changed = True

    # ---- bridge short silences ----
    while changed:
        changed = False
        result = []
        result_speech_content = []
        i = 0

        while i < len(bridged):
            if (
                i + 2 < len(bridged)
                and bridged[i]["type"] == "speech"
                and bridged[i + 1]["type"] == "non-speech"
                and bridged[i + 2]["type"] == "speech"
            ):
                gap = (bridged[i + 1]["end"] - bridged[i + 1]["start"]) / sr
                total_duration = bridged[i + 2]["end"] - bridged[i]["start"]
                total_duration_sec = total_duration / sr
                total_speech = speech_content[i] + speech_content[i + 2]
                speech_ratio = total_speech / total_duration

                if (
                    gap <= max_merge_gap
                    and speech_ratio >= min_speech_ratio
                    and total_duration_sec <= max_segment_duration
                ):
                    result.append(
                        {"start": bridged[i]["start"], "end": bridged[i + 2]["end"], "type": "speech"}
                    )
                    result_speech_content.append(
                        speech_content[i] + speech_content[i + 1] + speech_content[i + 2]
                    )
                    i += 3
                    changed = True
                    continue

            result.append(bridged[i])
            result_speech_content.append(speech_content[i])
            i += 1

        bridged = result
        speech_content = result_speech_content

    # ---- enforce min speech ----
    for seg in bridged:
        dur = (seg["end"] - seg["start"]) / sr
        if seg["type"] == "speech" and dur < min_speech_duration:
            seg["type"] = "non-speech"

    # ---- merge consecutive non-speeches ----
    merged = [bridged[0].copy()]
    for nxt in bridged[1:]:
        if merged[-1]["type"] == "non-speech" and nxt["type"] == "non-speech":
            merged[-1]["end"] = nxt["end"]
        else:
            merged.append(nxt.copy())

    # ---- remove short silences ----
    cleaned = [
        seg
        for seg in merged
        if not (seg["type"] == "non-speech" and (seg["end"] - seg["start"]) / sr < min_non_speech_duration)
    ]

    # ---- split long segments ----
    max_samp = int(max_segment_duration * sr)
    overlap_samp = int(overlap * sr)

    def split_long(seg: dict) -> list[dict]:
        start, end, typ = seg["start"], seg["end"], seg["type"]
        if end - start <= max_samp:
            return [seg]

        if typ == "speech":
            step_back = overlap_samp
            min_tail_samp = min_speech_duration * sr
        else:
            step_back = 0
            min_tail_samp = min_non_speech_duration * sr

        chunks = []
        pos = start
        while pos < end:
            chunk_end = pos + max_samp
            if chunk_end >= end:
                chunks.append({"start": pos, "end": end, "type": typ})
                break
            if (end - chunk_end) + step_back < min_tail_samp:
                chunks.append({"start": pos, "end": end, "type": typ})
                break
            chunks.append({"start": pos, "end": chunk_end, "type": typ})
            pos = chunk_end - step_back
        return chunks

    output = []
    for seg in cleaned:
        output.extend(split_long(seg))

    return output