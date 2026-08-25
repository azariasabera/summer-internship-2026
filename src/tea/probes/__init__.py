# src/tea/probes/__init__.py

"""Representation probes on top of a frozen MTKD student.

| Module | Ported from |
|---|---|
| `child_speech.py` | `probe_related.txt` |
| `feature_fusion.py` | `probe_related2.txt` (unique experiment logic only -- the data-loading it also contained, confirmed duplicated twice within that one file, is `tea.features.build_master_table`) |
"""

from tea.probes.child_speech import ChildSpeechProbe, load_dataset, probe_child_speech_cli
from tea.probes.feature_fusion import get_feature_combinations, get_feature_groups, probe_feature_fusion_cli, run_experiment

__all__ = [
    "ChildSpeechProbe", "load_dataset", "probe_child_speech_cli",
    "get_feature_groups", "get_feature_combinations", "run_experiment", "probe_feature_fusion_cli",
]