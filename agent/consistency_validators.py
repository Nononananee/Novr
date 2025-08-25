"""
Compatibility shim that re-exports fixed consistency validators used in tests.
"""
from .validation.consistency_validators_fixed import (
    run_all_validators,
    fact_check_validator,
    behavior_consistency_validator,
    dialogue_style_validator,
    trope_detector_validator,
    timeline_consistency_validator,
)

__all__ = [
    "run_all_validators",
    "fact_check_validator",
    "behavior_consistency_validator",
    "dialogue_style_validator",
    "trope_detector_validator",
    "timeline_consistency_validator",
]
