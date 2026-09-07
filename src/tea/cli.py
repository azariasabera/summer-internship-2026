# src.tea.cli.py

"""
Top-level command-line interface for Teacher Emotion Analysis (tea).

Commands are listed in **pipeline dependency order**. Run ``tea --help`` to
see the full list, or read ``doc/pipeline.md`` for the recommended sequence.

Examples
--------
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
import warnings
import textwrap
from collections.abc import Sequence
from typing import Final
from importlib.metadata import version

from tea.utils.config import load_config
from tea.utils.logging import get_logger, make_run_dir, save_run_info, tee_stdout_stderr
from tea.utils.paths import resolve, ensure_dir

# Type aliases used to describe entries in the pipeline command definitions.
CommandName = str
ModulePath = str
FunctionName = str
CommandDescription = str


# Pipeline stages 
_STAGE1: Final[list[tuple[CommandName, ModulePath, FunctionName, CommandDescription]]] = [
    ("chunk", "tea.vad", "chunk", "VAD segmentation -> generated/chunks + annotation CSVs"),
    ("merge-annotations", "tea.utils.io", "merge_annotations", "Copy gt_label / confidence / overlap from prepared CSVs into generated annotations"),
    ("apply-asr", "tea.asr", "transcribe_annotation_root", "Whisper transcribe + translate on speech chunks"),
    ("denoise", "tea.noise", "denoise", "DeepFilterNet or spectral subtraction (optional)"),
    ("extract-noise", "tea.noise.extraction", "extract_noise", "Build non-speech noise pool from annotated videos"),
    ("sentiment", "tea.features.sentiment", "sentiment_cli", "FI/EN text-sentiment probabilities from transcripts"),
]

_STAGE2: Final[list[tuple[CommandName, ModulePath, FunctionName, CommandDescription]]] = [
    ("train-teacher", "tea.teachers", "train_teacher", "Monolingual teacher fine-tune (Triton)"),
    ("train-mtkd", "tea.mtkd.train", "train_mtkd_cli", "Multilingual MTKD student train (Triton)"),
    ("finetune-classroom", "tea.classroom.finetune", "finetune_classroom_cli", "LOTO classroom fine-tune (Triton)"),
]


_STAGE3: Final[list[tuple[CommandName, ModulePath, FunctionName, CommandDescription]]] = [
    ("evaluate-mtkd", "tea.mtkd.evaluate", "evaluate_mtkd_cli", "Evaluate a student checkpoint on held-out benchmark splits"),
    ("calibrate", "tea.mtkd.calibrate", "calibrate_cli", "Temperature / bias calibration of a student checkpoint"),
]

_STAGE4: Final[list[tuple[CommandName, ModulePath, FunctionName, CommandDescription]]] = [
    ("infer-mtkd", "tea.mtkd.infer", "infer_mtkd_cli", "MTKD student inference -> pred_label / scores on CSVs"),
    ("extract-embeddings", "tea.mtkd.embeddings", "extract_embeddings_cli", "Pooled WavLM embeddings for probes (optional)"),
]

_STAGE5: Final[list[tuple[CommandName, ModulePath, FunctionName, CommandDescription]]] = [
    ("evaluate-classroom", "tea.analysis", "evaluate_classroom_cli", "WAR / UAR / confusion + child-speech / confidence breakdowns"),
    ("temporal", "tea.analysis", "temporal_cli", "Temporal consistency scores + smoothed emotion arcs"),
    ("noise-analysis", "tea.analysis", "noise_analysis_cli", "Noise filtering / augmentation distribution tables"),
    ("acoustic-by-emotion", "tea.analysis.acoustic_by_emotion", "acoustic_by_emotion_cli", "Acoustic feature box-plots by predicted emotion"),
    ("emotion-arc", "tea.analysis", "emotion_arc_per_video", "Emotion arc plots per video (smoothed)"),
]

_STAGE6: Final[list[tuple[CommandName, ModulePath, FunctionName, CommandDescription]]] = [
    ("confidence", "tea.confidence", "confidence_cli", "Binary / TCP / instance-temperature reliability"),
    ("probe-child-speech", "tea.probes.child_speech", "probe_child_speech_cli", "Child-speech logistic probe on embeddings"),
    ("probe-feature-fusion", "tea.probes.feature_fusion", "probe_feature_fusion_cli", "Handcrafted + embedding feature-fusion tables"),
]


_ALL_STAGES: Final[list[tuple[str, list[tuple[CommandName, ModulePath, FunctionName, CommandDescription]]]]] = [
    ("1. Data preparation", _STAGE1),
    ("2. Training (Triton / GPU)", _STAGE2),
    ("3. Checkpoint evaluation & calibration", _STAGE3),
    ("4. Inference (frozen checkpoints)", _STAGE4),
    ("5. Analysis / plots / tables", _STAGE5),
    ("6. Probes / confidence estimation", _STAGE6),
]

CLI_COMMANDS: Final[dict[CommandName, tuple[ModulePath, FunctionName]]] = {
    name: (mod, fn) for _, stage in _ALL_STAGES for name, mod, fn, _ in stage
}

COMMAND_HELP: Final[dict[CommandName, CommandDescription]] = {
    name: help_ for _, stage in _ALL_STAGES for name, _, _, help_ in stage
}

def build_help_epilog() -> str:
    """Build the detailed command list shown at the end of `tea --help`."""
    lines = [
        "Commands grouped by pipeline stage (top -> bottom reflects typical dependency order,",
        "not a strict requirement. See doc/pipeline.md for what each step actually needs):",
        "",
    ]

    for stage_title, stage_commands in _ALL_STAGES:
        lines.append(f"  {stage_title}")

        for command_name, _, _, description in stage_commands:
            lines.append(f"    tea {command_name:<22}  {description}")

        lines.append("")

    lines.append("See doc/pipeline.md for inputs/outputs of each step.")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser for the `tea` CLI."""
    parser = argparse.ArgumentParser(
        prog="tea",
        description="Teacher Emotion Analysis: classroom SER pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(build_help_epilog()),
    )

    parser.add_argument("--version", action="version", version=f"tea {version('tea')}")

    subparsers = parser.add_subparsers(dest="command", title="commands", metavar="<command>")

    for command_name, help_text in COMMAND_HELP.items():
        command_parser = subparsers.add_parser(command_name, help=help_text, description=help_text)

        command_parser.add_argument(
            "hydra_overrides",
            nargs="*",
            metavar="KEY=VALUE",
            help="Hydra overrides, e.g. paths.audio_root=/data/audio",
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and dispatch the selected command.

    Parameters
    ----------
    argv:
        Command-line arguments. If `None`, arguments are read from `sys.argv`.

    Returns
    -------
    int
        Exit status code returned by the selected command.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return dispatch_command(args.command, args.hydra_overrides)


def dispatch_command(command: str, hydra_overrides: Sequence[str]) -> int:
    """Load configuration, prepare run logging, and execute a command.

    Parameters
    ----------
    command:
        Name of the CLI command to run.
    hydra_overrides:
        Hydra `KEY=VALUE` overrides supplied on the command line.

    Returns
    -------
    int
        Exit status code of the executed command.
    """
    if command not in CLI_COMMANDS:
        print(f"[tea] Unknown command: {command}", file=sys.stderr)
        return 2
    
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

    # Show the command and overrides before logging starts.
    print(f"[tea] Command: {command}")
    if hydra_overrides:
        print("[tea] Hydra overrides:")
        for override in hydra_overrides:
            print(f"      {override}")
    else:
        print("[tea] Hydra overrides: none")

    # Load the Hydra configuration.
    cfg = load_config(hydra_overrides)

    # Create a directory for this run.
    logs_root = ensure_dir(resolve(cfg.paths.get("logs_root", "logs")))
    run_dir = make_run_dir(logs_root, command)

    # Save the exact command line used for this run.
    (run_dir / "command.txt").write_text(
        f"tea {command} {' '.join(hydra_overrides)}\n", encoding="utf-8",
    )

    logger = get_logger("tea.cli")

    # Capture all output produced from this point onward.
    with tee_stdout_stderr(run_dir / "stdout.log"):
        save_run_info(logger, cfg, run_dir)

        module_path, function_name = CLI_COMMANDS[command]

        try:
            module = importlib.import_module(module_path)
            command_func = getattr(module, function_name)
        except (ImportError, AttributeError) as exc:
            print(
                f"[tea] Failed to load {module_path}.{function_name}: {exc}\n"
                f"      Is the module implemented and installed (`pip install -e .`)?",
                file=sys.stderr,
            )
            return 1

        result = command_func(cfg=cfg)

        return int(result) if result is not None else 0


if __name__ == "__main__":
    sys.exit(main())
