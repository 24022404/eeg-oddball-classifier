"""
EEG Oddball Classifier Package
"""

__version__ = "0.1.0"

from .preprocessing import preprocess_subject_eeg, run_all_subjects

__all__ = [
    "preprocess_subject_eeg",
    "run_all_subjects",
]