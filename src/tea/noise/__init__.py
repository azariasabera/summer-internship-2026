"""Noise processing, denoising, and noise augmentation."""

from __future__ import annotations

from omegaconf import DictConfig

from tea.noise.denoiser import Denoiser
from tea.noise.spectral_subtraction import SpectralSubtractor, build_noise_reference
from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir, resolve

logger = get_logger(__name__)


def denoise(cfg: DictConfig) -> int:
    """`tea denoise`: run the configured denoising method.

    Parameters
    ----------
    cfg:
        Resolved Hydra config. `cfg.noise.method` selects `"deepfilter"`
        (default) or `"spectral"`.
    """
    import json
    import librosa
    import soundfile as sf

    method = cfg.noise.get("method", "deepfilter")

    if method not in ["deepfilter", "spectral"]:
        raise ValueError(f"Unknown noise.method='{method}'")

    json_annotations = resolve(cfg.noise.json_annotations) # contains per-video json meta data

    out_dir = ensure_dir(resolve(cfg.noise.get(method).get("save_dir")))

    json_files = sorted(json_annotations.glob("*.json"))
    logger.info("Denoising %d file(s) with method=%s -> %s", len(json_files), method, out_dir)

    if method == "deepfilter":
        # add cfg.noise.deepfilter.atten_lim_db to out_dir to save the atten_lim_db value in the output directory name
        out_dir = ensure_dir(out_dir / f"atten_lim_db_{cfg.noise.deepfilter.atten_lim_db}")
        denoiser = Denoiser(cfg)

        try:
            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                audio_path = resolve(meta["audio_path"])
                sample_rate = int(meta.get("sr", 16_000))

                video_out_dir = ensure_dir(out_dir / json_file.stem)

                waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

                for idx, segment in enumerate(meta.get("segments", [])):
                    start = int(segment["start"])
                    end = int(segment["end"])
                    segment_type = segment.get("type", "speech")

                    prefix = "s" if segment_type == "speech" else "n"
                    chunk_name = f"chunk_{prefix}_{idx}"

                    chunk = waveform[start:end]

                    # Temporary input for DeepFilterNet.
                    chunk_input = video_out_dir / f"{chunk_name}_input.wav"
                    chunk_output = video_out_dir / f"{chunk_name}.wav"

                    sf.write(chunk_input, chunk, sample_rate)

                    denoiser.enhance(
                        input_path=chunk_input,
                        output_path=chunk_output,
                        save=True,
                        atten_lim_db=cfg.noise.deepfilter.atten_lim_db,
                    )

                    chunk_input.unlink()

                logger.info("  %s: denoised %d chunks", json_file.stem, len(meta.get("segments", [])))

        finally:
            denoiser.close()

    elif method == "spectral":

        from tea.utils.io import load_annotation_csvs
        import numpy as np

        videos_df = load_annotation_csvs(
            annotation_root=cfg.noise.csv_annotations,
            exclude=None,
            add_audio_path=True,
            json_dir=cfg.noise.json_annotations,
        )
        subtractor = SpectralSubtractor(
            n_fft=int(cfg.noise.spectral.get("n_fft", 1024)),
            hop=int(cfg.noise.spectral.get("hop", 256)),
            alpha=float(cfg.noise.spectral.get("alpha", 1.0)),
            beta=float(cfg.noise.spectral.get("beta", 0.05)),
        )

        n_noise_chunks = int(cfg.noise.spectral.get("n_noise_chunks", 5))

        for video, video_df in videos_df.groupby("video", sort=True):

            logger.info("Processing %s", video)

            video_out_dir = ensure_dir(out_dir / video)

            reference = build_noise_reference(video_df, n_chunks=n_noise_chunks)

            if reference is None:
                logger.warning("%s: no gt_label-NaN noise chunks", video)
                continue

            noise_signal = reference["noise_signal"]
            noise_sr = reference["sr"]
            selected = reference["selected"]

            logger.info("%s: %d noise candidates, using %d", video, reference["n_candidates"], len(selected))

            for item in selected:
                logger.info("  %s: power=%.6e", item["name"], item["power"])

            logger.info("%s: selected noise median power=%.6e", video, np.median([item["power"] for item in selected]))

            audio_paths = video_df["audio_path"].dropna().unique()
            audio_path = resolve(audio_paths[0])

            waveform, sr = librosa.load(audio_path, sr=None, mono=True)
            if sr != noise_sr:
                raise ValueError(f"{video}: source sample rate={sample_rate}, noise sample rate={noise_sr}")

            # Applying the spectral subtraction
            for _, row in video_df.iterrows():
                start = int(row["start"])
                end = int(row["end"])

                chunk = waveform[start:end]

                y = subtractor.subtract(speech=chunk, noise=noise_signal)
                y = np.clip(y, -1.0, 1.0)

                chunk_out_file = video_out_dir / f"{row['name']}.wav"
                sf.write(chunk_out_file, y, sr)

            logger.info("%s: denoised %d chunks", video, len(video_df))
    else:
        logger.error("Unknown noise.method=%s", method)
        return 2

    return 0


# def denoise(cfg: DictConfig) -> int:
#     """Run the configured audio denoising pipeline.

#     Parameters
#     ----------
#     cfg:
#         Hydra configuration for the current run.

#     Returns
#     -------
#     int
#         Process exit status.
#     """
#     print("[tea] Running audio denoising")
#     print()
#     print("Noise configuration:")
#     print(OmegaConf.to_yaml(cfg.noise, resolve=True))

#     return 0


# def extract_noise(cfg: DictConfig) -> int:
#     """Extract noise segments from the benchmark data.

#     Parameters
#     ----------
#     cfg:
#         Hydra configuration for the current run.

#     Returns
#     -------
#     int
#         Process exit status.
#     """
#     print("[tea] Running noise extraction")
#     print()
#     print("Noise configuration:")
#     print(OmegaConf.to_yaml(cfg.noise, resolve=True))

#     return 0
