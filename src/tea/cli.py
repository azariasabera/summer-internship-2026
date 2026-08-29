"""
Top-level command-line interface for Teacher Emotion Analysis (tea).

Commands are listed in **pipeline dependency order**. Run ``tea --help`` to
see the full list, or read ``doc/pipeline.md`` for the recommended sequence.

Examples
--------
::

    tea chunk
    tea merge-annotations
    tea apply-asr
    tea infer-mtkd
    tea evaluate-classroom
"""

from __future__ import annotations

import argparse
import importlib
import sys
import textwrap
from collections.abc import Sequence
from typing import Final

from tea.utils.config import load_config

VERSION: Final[str] = "0.1.0"

# type aliases
COMMAND = str
MODULE = str
CLI_FUNCTION = str
DESCRIPTION = str

# ---------------------------------------------------------------------------
# Pipeline stages (order = dependency order for everyday reproduction)
# ---------------------------------------------------------------------------

_STAGE1: Final[list[tuple[COMMAND, MODULE, CLI_FUNCTION, DESCRIPTION]]] = [
    ("chunk", "tea.vad", "chunk", "VAD segmentation -> generated/chunks + annotation CSVs"),
    ("merge-annotations", "tea.utils.io", "merge_annotations", "Copy gt_label / confidence / overlap from prepared CSVs into generated annotations"),
    ("apply-asr", "tea.asr", "transcribe_annotation_root", "Whisper transcribe + translate on speech chunks"),
    ("denoise", "tea.noise", "denoise", "DeepFilterNet or spectral subtraction (optional)"),
    ("extract-noise", "tea.noise", "extract_noise", "Build non-speech noise pool from annotated videos"),
    ("sentiment", "tea.features", "sentiment_cli", "FI/EN text-sentiment probabilities from transcripts"),
]

_STAGE2: Final[list[tuple[COMMAND, MODULE, CLI_FUNCTION, DESCRIPTION]]] = [
    ("infer-mtkd", "tea.mtkd.infer", "infer_mtkd_cli", "MTKD student inference -> pred_label / scores on CSVs"),
    (
        "extract-embeddings",
        "tea.mtkd.embeddings",
        "extract_embeddings_cli",
        "Pooled WavLM embeddings for probes (optional)",
    ),
]

_STAGE3: Final[list[tuple[COMMAND, MODULE, CLI_FUNCTION, DESCRIPTION]]] = [
    (
        "evaluate-classroom",
        "tea.analysis",
        "evaluate_classroom_cli",
        "WAR / UAR / confusion + child-speech / confidence breakdowns",
    ),
    ("temporal", "tea.analysis", "temporal_cli", "Temporal consistency scores + smoothed emotion arcs"),
    ("noise-analysis", "tea.analysis", "noise_analysis_cli", "Noise filtering / augmentation distribution tables"),
    (
        "acoustic-by-emotion",
        "tea.analysis.acoustic_by_emotion",
        "acoustic_by_emotion_cli",
        "Acoustic feature box-plots by predicted emotion",
    ),
    ("confidence", "tea.confidence", "confidence_cli", "Binary / TCP / instance-temperature reliability"),
    ("probe-child-speech", "tea.probes", "probe_child_speech_cli", "Child-speech logistic probe on embeddings"),
    (
        "probe-feature-fusion",
        "tea.probes",
        "probe_feature_fusion_cli",
        "Handcrafted + embedding feature-fusion tables",
    ),
]

_STAGE4: Final[list[tuple[COMMAND, MODULE, CLI_FUNCTION, DESCRIPTION]]] = [
    ("train-teacher", "tea.teachers", "train_teacher", "Monolingual teacher fine-tune (Triton)"),
    ("train-mtkd", "tea.mtkd", "train_mtkd_cli", "Multilingual MTKD student train (Triton)"),
    (
        "finetune-classroom",
        "tea.classroom.finetune",
        "finetune_classroom_cli",
        "LOTO classroom fine-tune configs A-F (Triton)",
    ),
]

_STAGE5: Final[list[tuple[COMMAND, MODULE, CLI_FUNCTION, DESCRIPTION]]] = [
    (
        "evaluate-mtkd",
        "tea.mtkd.evaluate",
        "evaluate_mtkd_cli",
        "Evaluate a student checkpoint on held-out benchmark splits",
    ),
    ("calibrate", "tea.mtkd", "calibrate_cli", "Temperature / bias calibration of a student checkpoint"),
]

_ALL_STAGES: Final[list[tuple[str, list[tuple[COMMAND, MODULE, CLI_FUNCTION, DESCRIPTION]]]]] = [
    ("1. Data preparation", _STAGE1),
    ("2. Inference (frozen checkpoints)", _STAGE2),
    ("3. Analysis / probes / confidence", _STAGE3),
    ("4. Training (Triton / GPU)", _STAGE4),
    ("5. Checkpoint evaluation helpers", _STAGE5),
]

CLI_COMMANDS: Final[dict[COMMAND, tuple[MODULE, CLI_FUNCTION]]] = {
    name: (mod, fn) for _, stage in _ALL_STAGES for name, mod, fn, _ in stage
}

COMMAND_HELP: Final[dict[COMMAND, DESCRIPTION]] = {
    name: help_ for _, stage in _ALL_STAGES for name, _, _, help_ in stage
}

def _epilog() -> str:
    lines = [
        "Pipeline order (run top -> bottom for full classroom reproduction):",
        "",
    ]
    for title, stage in _ALL_STAGES:
        lines.append(f"  {title}")
        for name, _, _, help_ in stage:
            lines.append(f"    tea {name:<22}  {help_}")
        lines.append("")
    lines.append("See doc/pipeline.md for inputs/outputs of each step.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tea",
        description="Teacher Emotion Analysis -- classroom SER pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(_epilog()),
    )
    parser.add_argument("--version", action="version", version=f"tea {VERSION}")

    sub = parser.add_subparsers(dest="command", title="commands", metavar="<command>")

    for name, help_text in COMMAND_HELP.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument(
            "hydra_overrides",
            nargs="*",
            metavar="KEY=VALUE",
            help="Hydra overrides, e.g. paths.audio_root=/data/audio",
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return dispatch_command(args.command, args.hydra_overrides)


def dispatch_command(command: str, hydra_overrides: Sequence[str]) -> int:
    if command not in CLI_COMMANDS:
        print(f"[tea] Unknown command: {command}", file=sys.stderr)
        return 2

    print(f"[tea] Command: {command}")
    if hydra_overrides:
        print("[tea] Hydra overrides:")
        for o in hydra_overrides:
            print(f"      {o}")
    else:
        print("[tea] Hydra overrides: none")

    cfg = load_config(hydra_overrides)
    module_path, func_name = CLI_COMMANDS[command]

    try:
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
    except (ImportError, AttributeError) as exc:
        print(
            f"[tea] Failed to load {module_path}.{func_name}: {exc}\n"
            f"      Is the module implemented and installed (`pip install -e .`)?",
            file=sys.stderr,
        )
        return 1

    result = func(cfg=cfg)
    return int(result) if result is not None else 0


if __name__ == "__main__":
    sys.exit(main())
