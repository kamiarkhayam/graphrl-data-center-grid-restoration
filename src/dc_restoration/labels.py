"""Display-label compatibility; numerical columns and method identities stay intact."""

PUBLIC_LABELS = {
    "TRUE_GNN Harmonized A2C": "GraphRL-A2C",
    "TRUE_GNN Harmonized PPO": "GraphRL-PPO",
    "TRUE_GNN Harmonized BC": "GraphBC",
    "MLP Harmonized A2C": "MLP-A2C",
    "CNN Harmonized A2C": "CNN-A2C",
    "Greedy Harmonized": "Greedy",
    "Random top-K Harmonized": "Random Top-K",
    "True-GNN-A2C": "GraphRL-A2C",
    "True-GNN-PPO": "GraphRL-PPO",
    "True-GNN BC": "GraphBC",
    "MLP-RL": "MLP-A2C",
    "CNN-RL": "CNN-A2C",
    "Greedy VoLL": "Greedy",
    "Random top-K": "Random Top-K",
}


def public_display_columns(frame):
    """Normalize policy labels while retaining numerical and resource schema fields."""
    result = frame.copy()
    for column in ("policy", "display_policy", "display_name", "paper_policy", "baseline_policy"):
        if column in result:
            result[column] = result[column].replace(PUBLIC_LABELS)
    for column in ("resource_case", "dc_resource_display"):
        if column in result:
            result[column] = result[column].replace(
                {
                    "Finite-energy E12": "E12",
                    "Finite-energy E24": "E24",
                    "Power-bounded anchor": "Power-only reference",
                }
            )
    return result


def public_palette(methods):
    return {PUBLIC_LABELS.get(name, name): style for name, style in methods.items()}
