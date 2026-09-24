"""Edge-weighted DF-QuAD over the repository's unchanged BAG and influence classes."""

from Uncertainpy.src.uncertainpy.gradual.algorithms.Acyclic import computeTopOrder
from Uncertainpy.src.uncertainpy.gradual.semantics.modular.LinearInfluence import LinearInfluence
from Uncertainpy.src.uncertainpy.gradual.semantics.modular.ProductAggregation import ProductAggregation


def edge_key(kind, source, target):
    return (kind, source, target)


def existing_edges(bag):
    for edge in bag.supports:
        yield edge_key("support", edge.supporter.name, edge.supported.name)
    for edge in bag.attacks:
        yield edge_key("attack", edge.attacker.name, edge.attacked.name)


def compute_ew_dfquad(bag, edge_weights):
    """Replace each parent strength with strength times its own edge weight."""
    expected = set(existing_edges(bag))
    if set(edge_weights) != expected:
        raise ValueError("Edge weights must cover exactly the existing graph edges")
    for weight in edge_weights.values():
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Edge weights must lie in [0, 1]")

    order = computeTopOrder(bag)
    if order is None:
        raise ValueError("EW-DF-QuAD requires an acyclic graph")

    aggregation = ProductAggregation()
    influence = LinearInfluence(conservativeness=1)
    strengths = {argument: argument.initial_weight for argument in order}

    for target in order:
        weighted = {}
        for parent in target.attackers:
            weighted[parent] = strengths[parent] * edge_weights[
                edge_key("attack", parent.name, target.name)
            ]
        for parent in target.supporters:
            weighted[parent] = strengths[parent] * edge_weights[
                edge_key("support", parent.name, target.name)
            ]
        aggregate = aggregation.aggregate_strength(
            target.attackers, target.supporters, weighted
        )
        strength = influence.compute_strength(target.initial_weight, aggregate)
        target.strength = strength
        strengths[target] = strength
    return strengths
