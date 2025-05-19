# Utilities for assigning curators to pending reports.
#
# This module implements weighted random assignment of curators based on
# stake and reputation, as well as recording their hidden votes.

import random
from typing import Dict, Any, List

# Weight multiplier for positive reputation when selecting curators.
BETA_REP_WEIGHT = 0.1

# Number of curators to assign to each report.
N_CURATORS_PER_CASE = 7


def weight(agent: Dict[str, Any]) -> float:
    # Return the selection weight for a curator agent.
    return agent["stake"] * (1 + BETA_REP_WEIGHT * max(0, agent["rep"]))


def assign_curators(params: Dict[str, Any], step: int, sL: List[Any], s: Dict[str, Any]) -> Dict[str, Any]:
    # Assign curators to pending reports and record hidden votes.
    rng: random.Random = params["rng"]
    for rep in s["reports_pending"]:
        population = s["curators"]
        weights = [weight(c) for c in population]
        chosen = rng.choices(population, weights, k=N_CURATORS_PER_CASE)
        rep["curators"] = chosen
        for c in chosen:
            acc = 0.55 if c["rep"] < 0 else 0.70 if c["rep"] < 5 else 0.90
            vote = rep["truth"] if rng.random() < acc else (not rep["truth"])
            rep["votes"].append({"agent": c, "vote": vote})
    return {}
