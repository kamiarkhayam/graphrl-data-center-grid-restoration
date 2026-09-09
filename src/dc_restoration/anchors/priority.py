"""The allocation mode used by the archived final restoration trajectories."""

FIXED_PRIORITY_COMPATIBILITY = "critical_first_fixed_priority"


def validate_allocation_mode(mode: str) -> str:
    """Do not silently reinterpret an archived controller as a new dispatch objective."""
    if mode != FIXED_PRIORITY_COMPATIBILITY:
        raise ValueError(
            "Only critical_first_fixed_priority is supported: deterministic critical-first "
            "allocation with a fixed within-class priority."
        )
    return mode
