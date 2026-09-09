"""The eight reported methods and their final configuration links."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    architecture: str
    algorithm: str | None
    configuration: str
    archive_label: str


METHODS = {
    "GraphRL-A2C": Method("true_gnn", "a2c", "graphrl_a2c", "TRUE_GNN Harmonized A2C"),
    "GraphRL-PPO": Method("true_gnn", "ppo", "graphrl_ppo", "TRUE_GNN Harmonized PPO"),
    "GraphBC": Method("true_gnn", "bc", "graphbc", "TRUE_GNN Harmonized BC"),
    "MLP-A2C": Method("mlp", "a2c", "mlp_a2c", "MLP Harmonized A2C"),
    "CNN-A2C": Method("cnn", "a2c", "cnn_a2c", "CNN Harmonized A2C"),
    "Greedy": Method("greedy", None, "greedy", "Greedy Harmonized"),
    "Random Top-K": Method("random", None, "random_top_k", "Random top-K Harmonized"),
    "Bounded GA": Method("bounded_ga", None, "bounded_ga", "bounded_ga_source_duration_damage"),
}

TRAINABLE = tuple(name for name, method in METHODS.items() if method.algorithm is not None)
EVALUATED = tuple(name for name in METHODS if name != "Bounded GA")
