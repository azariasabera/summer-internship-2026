"""
Top-level command-line interface for the Teacher Emotion Analysis (tea) project.

The `tea` command provides a stable interface for running pipeline stages
and passing Hydra configuration overrides.

Examples
--------
Run a command with default configuration:

    tea chunk

Override Hydra configuration values:

    tea chunk vad.overlap=1.0
    tea chunk vad.max_segment_duration=8.0

Show the available commands:

    tea --help
"""

from __future__ import annotations

import argparse
import sys
import importlib
from collections.abc import Sequence
from typing import Final

from tea.utils.config import load_config


VERSION: Final[str] = "0.1.0"


# Mapping from CLI command names to the module containing the implementation.
COMMAND_MODULES: Final[dict[str, str]] = {
    "chunk": "vad",
    "denoise": "noise",
    "extract-noise": "noise",
    "sentiment": "asr",
    "infer-mtkd": "mtkd",
    "evaluate-classroom": "analysis",
    "confidence": "confidence",
    "probe-child-speech": "probes",
    "probe-feature-fusion": "probes",
    "noise-analysis": "analysis",
    "temporal": "analysis",
}

# command name -> (module path, function name). Every implemented command
# must have an entry here; dispatch_command uses this instead of an
# if/elif chain so adding a command is one line, not a new branch.
COMMAND_FUNCTIONS: Final[dict[str, tuple[str, str]]] = {
    "chunk": ("tea.vad", "chunk"),
    "denoise": ("tea.noise", "denoise"),
    "extract-noise": ("tea.noise", "extract_noise"),
}


COMMAND_HELP: Final[dict[str, str]] = {
    "chunk": "VAD-based segmentation and optional denoising",
    "denoise": "DeepFilterNet / spectral subtraction on audio",
    "extract-noise": "Extract non-speech noise chunk paths from annotated videos",
    "sentiment": "Extract FI/EN text-sentiment probabilities",
    "infer-mtkd": "Run MTKD student inference on classroom chunks",
    "evaluate-classroom": "Classroom WAR/UAR/confusion and breakdowns",
    "confidence": "Binary / TCP / instance-temperature confidence",
    "probe-child-speech": "Child-speech logistic probe on embeddings",
    "probe-feature-fusion": "Handcrafted feature fusion experiments",
    "noise-analysis": "Noise filtering and augmentation distribution tables",
    "temporal": "Temporal consistency and smoothed emotion arcs",
}


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level `tea` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for the project CLI.
    """
    parser = argparse.ArgumentParser(
        prog="tea",
        description="Teacher Emotion Analysis: classroom SER pipeline",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"tea {VERSION}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="<command>",
    )

    for command, help_text in COMMAND_HELP.items():
        subparser = subparsers.add_parser(
            command,
            help=help_text,
            description=help_text,
        )

        subparser.add_argument(
            "hydra_overrides",
            nargs="*",
            metavar="KEY=VALUE",
            help="Hydra configuration overrides, e.g. vad.overlap=1.5",
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `tea` command-line interface.

    Parameters
    ----------
    argv:
        Command-line arguments. If `None`, arguments are read from `sys.argv`.

    Returns
    -------
    int
        Process exit status. `0` indicates success.
    """
    parser = build_parser()
    args: argparse.Namespace = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return dispatch_command(
        command=args.command,
        hydra_overrides=args.hydra_overrides,
    )


def dispatch_command(command: str, hydra_overrides: Sequence[str]) -> int:
    """Dispatch a CLI command to its implementation.

    Parameters
    ----------
    command:
        Name of the pipeline command, such as "chunk".
    hydra_overrides:
        Hydra-style configuration overrides such as ["vad.overlap=1.5"].

    Returns
    -------
    int
        Process exit status.

    Notes
    -----
    The actual pipeline implementations are not wired yet. This function
    currently provides the dispatch boundary that will later connect the
    CLI to the corresponding module.
    """
    module = COMMAND_MODULES.get(command)

    if module is None:
        print(
            f"[tea] Unknown command: {command}",
            file=sys.stderr,
        )
        return 2

    print(f"[tea] Command: {command}")
    print(f"[tea] Module: src/tea/{module}/")

    if hydra_overrides:
        print("[tea] Hydra overrides:")
        for override in hydra_overrides:
            print(f"      {override}")
    else:
        print("[tea] Hydra overrides: none")


    cfg = load_config(hydra_overrides)

    entry = COMMAND_FUNCTIONS.get(command)
    if entry is None:
        print()
        print(f"[tea] Command '{command}' is registered but not yet implemented.")
        return 0

    module_path, func_name = entry
    func = getattr(importlib.import_module(module_path), func_name)
    return func(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())