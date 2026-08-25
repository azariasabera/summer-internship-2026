# src/tea/analysis/__init__.py

"""Post-hoc analysis of MTKD predictions on classroom recordings."""

from tea.analysis.acoustic_by_emotion import acoustic_by_emotion_cli, add_loudness, add_speaking_rate, summarize_by_predicted_emotion
from tea.analysis.classroom import evaluate_classroom_cli, evaluate_per_video_and_overall, evaluate_predictions, join_predictions
from tea.analysis.noise import class_distribution, compare_conditions, noise_analysis_cli, non_speech_prediction_distribution
from tea.analysis.temporal import (
    consistency_report, gaussian_smooth, moving_average, plot_emotion_arc, process_video_stability, stability_test,
    temporal_cli, temporal_consistency, emotion_arc_per_video
)

__all__ = [
    "join_predictions", "evaluate_predictions", "evaluate_per_video_and_overall", "evaluate_classroom_cli",
    "add_loudness", "add_speaking_rate", "summarize_by_predicted_emotion", "acoustic_by_emotion_cli",
    "temporal_consistency", "consistency_report", "process_video_stability", "stability_test",
    "moving_average", "gaussian_smooth", "plot_emotion_arc", "temporal_cli", "emotion_arc_per_video",
    "class_distribution", "compare_conditions", "non_speech_prediction_distribution", "noise_analysis_cli",
]
