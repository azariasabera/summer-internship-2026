# src/tea/vad/video.py

"""Utilities for generating video chunk commands."""

from __future__ import annotations

from pathlib import Path

from tea.utils.paths import ensure_dir

from typing import Union

def generate_video_chunk_commands(
    segment_list: list[dict], video_paths: list[str], out_dir: Union[str, Path] = "chunks", run: bool = True
) -> list[dict]:
    """Generate (and optionally print) ffmpeg commands to cut video chunks matching audio segments.

    Matches videos to segment metadata by filename stem. Does not itself
    shell out to ffmpeg by default (`run=True` only prints the commands, it
    does not execute them) -- this mirrors the original notebook usage,
    where commands were reviewed before running.

    Parameters
    ----------
    segment_list:
        Output of `Segmenter.chunk_vad` / `chunk_fixed`.
    video_paths:
        Video file paths to match against `segment_list` by stem.
    out_dir:
        Root directory for `<audio_stem>_video/chunk_*.mp4`.
    run:
        If True, print each ffmpeg command as it's generated.

    Returns
    -------
    list[dict]
        One `{"audio_stem", "video_path", "commands"}` dict per matched video.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    video_map = {Path(v).stem: v for v in video_paths}
    all_cmds = []

    for item in segment_list:
        audio_stem = Path(item["audio_path"]).stem
        if audio_stem not in video_map:
            continue

        video_path = video_map[audio_stem]
        sr = item["sr"]
        cmds = []

        video_chunk_dir = ensure_dir(out_dir / f"{audio_stem}_video")

        for idx, seg in enumerate(item["segments"]):
            start = seg["start"] / sr
            end = seg["end"] / sr
            typ_short = "n" if seg["type"] == "non-speech" else "s"
            fname = f"chunk_{typ_short}_{idx}.mp4" if seg["type"] is not None else f"chunk_{idx}.mp4"
            out_path = video_chunk_dir / fname

            cmd = f'ffmpeg -y -i "{video_path}" -ss {start:.3f} -to {end:.3f} -c copy "{out_path}"'
            cmds.append(cmd)
            if run:
                print(cmd)

        all_cmds.append({"audio_stem": audio_stem, "video_path": video_path, "commands": cmds})
        if run:
            print("\n" + "=" * 60 + "\n")

    return all_cmds