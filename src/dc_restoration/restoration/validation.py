"""Input contracts around the preserved restoration simulator."""

from collections.abc import Iterable


def validate_repair_sequence(damaged: Iterable[str], sequence: Iterable[str]) -> list[str]:
    """Reject repeated or undamaged repairs; partial restoration orders are valid."""
    remaining = set(map(str, damaged))
    order = list(map(str, sequence))
    for edge in order:
        if edge not in remaining:
            raise ValueError(f"Repair must name an unrepaired damaged edge: {edge}")
        remaining.remove(edge)
    return order


def validate_case(model) -> None:
    """Validate identifier relationships required by the reported implementation."""
    nodes = set(model.nodes["node_id"].astype(str))
    edges = set(model.edges["edge_id"].astype(str))
    if len(nodes) != len(model.nodes) or len(edges) != len(model.edges):
        raise ValueError("Node and edge identifiers must be unique.")
    if model.base.source_node not in nodes:
        raise ValueError("The configured source node is missing.")
    if (
        not set(model.edges["from_node"].astype(str)) | set(model.edges["to_node"].astype(str))
        <= nodes
    ):
        raise ValueError("Edge endpoints must refer to existing nodes.")
    if not set(model.loads["node_id"].astype(str)) <= nodes:
        raise ValueError("Loads must refer to existing nodes.")
    if not set(model.damage["edge_id"].astype(str)) <= edges:
        raise ValueError("Damage must refer to existing edges.")
    if model.damage.duplicated(["scenario_key", "edge_id"]).any():
        raise ValueError("Damage contains repeated edge identifiers within a scenario.")
    if (model.damage["repair_time_hours"] <= 0).any():
        raise ValueError("Repair durations must be positive.")
    if any(value < 0 for value in model.dc_support_by_anchor.values()):
        raise ValueError("Anchor support caps must be nonnegative.")
