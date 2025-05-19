# Report processing logic for Skygate.
#
# This module defines a helper function ``process_reports`` which is
# responsible for applying report outcomes and distributing rewards.
# The code is based on the snippet labelled "SKYGATE – CHUNK 4".
#
# The constants below are simplistic stand-ins. In a full implementation
# these would likely be provided by other modules or configuration.

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Placeholder configuration constants
# ---------------------------------------------------------------------------
# Fraction of ``EMISSION_PER_REPORT`` allocated to different parties when a
# report is processed. These values should sum to ``1.0`` in practice.
REWARD_SPLIT = {
    "witness": 0.2,
    "curator": 0.5,
    "treasury": 0.3,
}

# Tokens minted for each verified report. This is a simplified stand‑in for
# demonstration purposes only.
EMISSION_PER_REPORT = 10.0

# Amount of stake associated with each report. Used when applying slashing
# penalties to incorrect curators.
STAKE_PER_REPORT = 5.0

# Fraction of ``STAKE_PER_REPORT`` slashed from a curator when they vote
# incorrectly.
SLASH_FRACTION = 0.5


# ---------------------------------------------------------------------------
# ``process_reports`` implementation
# ---------------------------------------------------------------------------

def process_reports(params: Dict[str, Any], step: int, sL: Dict[str, Any], s: Dict[str, Any]) -> Dict[str, Any]:
    # Process all pending reports and update global state.
    #
    # Parameters
    # ----------
    # params : dict
    #     Simulation parameters (unused here but kept for API compatibility).
    # step : int
    #     Current timestep of the simulation.
    # sL : dict
    #     Previous state (unused here but part of cadCAD's API).
    # s : dict
    #     Mutable simulation state containing a ``reports_pending`` list along
    #     with token and staking information.
    #
    # Returns
    # -------
    # dict
    #     Updated state ``s`` after processing all pending reports.

    verified, rejected = 0, 0
    treasury_inflow = 0.0
    new_tokens = 0.0

    # Iterate over all reports that are pending evaluation
    for rep in s.get("reports_pending", []):
        yes = sum(v["vote"] for v in rep.get("votes", []))
        no = len(rep.get("votes", [])) - yes
        majority_truth = yes > no
        outcome_is_verified = majority_truth

        # --- witness outcome ---
        witness_reward = 0.0
        if outcome_is_verified and rep.get("truth"):
            verified += 1
            witness_reward = REWARD_SPLIT["witness"] * EMISSION_PER_REPORT
            new_tokens += EMISSION_PER_REPORT
        else:
            rejected += 1
            treasury_inflow += rep.get("deposit", 0.0)  # deposit confiscated

        # --- per-curator accounting ---
        majority_flag = rep.get("truth") if outcome_is_verified else (not rep.get("truth"))
        correct_curators = [v["agent"] for v in rep.get("votes", []) if v["vote"] == majority_flag]
        wrong_curators = [v["agent"] for v in rep.get("votes", []) if v["vote"] != majority_flag]

        # rewards to correct curators
        if correct_curators:
            curator_share = REWARD_SPLIT["curator"] * EMISSION_PER_REPORT
            reward_each = curator_share / len(correct_curators)
            for c in correct_curators:
                c["stake"] += reward_each
                c["rep"] += 1

        # slashing wrong curators
        for c in wrong_curators:
            penalty = SLASH_FRACTION * STAKE_PER_REPORT
            c["stake"] -= penalty
            treasury_inflow += penalty
            c["rep"] -= 1

        # treasury emission slice
        treasury_inflow += REWARD_SPLIT["treasury"] * EMISSION_PER_REPORT
        if not outcome_is_verified:
            # if rejected, skip witness 20 %
            new_tokens += EMISSION_PER_REPORT * (
                REWARD_SPLIT["curator"] + REWARD_SPLIT["treasury"]
            )

    # -------- global state updates --------
    s["S_total"] = s.get("S_total", 0.0) + new_tokens
    s["S_treasury"] = s.get("S_treasury", 0.0) + treasury_inflow
    s["reports_pending"] = []  # cleared
    s["month_stats"] = dict(verified=verified, rejected=rejected, minted=new_tokens)
    return s

