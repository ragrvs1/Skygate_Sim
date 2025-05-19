# Exogenous process for report arrivals.
#
# This module implements the monthly arrival of witness reports based on a
# demand curve.

from math import ceil
from typing import List, Dict, Any

# Default growth rate used in the demand curve R(t) = 100 * (1+g)^t
GROWTH_RATE: float = 0.05

# Default deposit required for each witness report
WITNESS_DEPOSIT: int = 10


def report_arrival(params: Dict[str, Any], step: int, sL: List[dict], s: dict) -> Dict[str, List[Dict[str, Any]]]:
    # Generate new witness reports for the current timestep.
    #
    # Parameters
    # ----------
    # params : dict
    #     Parameters provided by cadCAD, must include a random number generator
    #     under ``rng`` and a ``truth_ratio`` controlling the fraction of
    #     legitimate reports.
    # step : int
    #     Current cadCAD step (unused).
    # sL : list
    #     State history (unused).
    # s : dict
    #     Current state dictionary. Expects ``t`` indicating the month index.
    #
    # Returns
    # -------
    # dict
    #     Dictionary with key ``"new_reports"`` mapping to the list of generated
    #     reports.
    t = s["t"]
    R_t = ceil(100 * (1 + GROWTH_RATE) ** t)  # eq (1)
    new_reports = [
        {
            "deposit": WITNESS_DEPOSIT,
            "truth": params["rng"].random() < params["truth_ratio"],
            "curators": [],
            "votes": [],
        }
        for _ in range(R_t)
    ]
    return {"new_reports": new_reports}


def update_pending_reports(params: Dict[str, Any], step: int, sL: List[dict], s: dict, _input: dict) -> dict:
    # Append newly arrived reports to the pending queue.
    s["reports_pending"].extend(_input["new_reports"])
    return s
