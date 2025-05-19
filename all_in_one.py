# Combined file containing code from all modules

# ===== File: README.md =====

# Skygate Simulation

This repository contains a simple **cadCAD** simulation that models token emissions and APR rewards for different stakeholder groups. The simulation is implemented in a single Python script: `sim.py`.

## Requirements

The project requires Python 3.11 or newer. Install dependencies via `pip`:

```bash
pip install cadCAD pandas numpy matplotlib
```

## Running the Simulation

Execute the simulation by running:

```bash
python sim.py
```

The script will run multiple Monte Carlo simulations and generate two plots:

1. **Emission Pool Over Time** – average remaining tokens in the emission pool.
2. **Cumulative Reports Over Time** – average number of reports submitted.

## Project Structure

- `sim.py` – defines the simulation configuration, policies, and plotting utilities.
- `README.md` – project overview and usage instructions.

Feel free to modify parameters in `sim.py` (such as the number of reports per step) to explore different scenarios.

# ===== File: curation.py =====

"""Utilities for assigning curators to pending reports.

This module implements weighted random assignment of curators based on
stake and reputation, as well as recording their hidden votes.
"""

import random
from typing import Dict, Any, List

# Weight multiplier for positive reputation when selecting curators.
BETA_REP_WEIGHT = 0.1

# Number of curators to assign to each report.
N_CURATORS_PER_CASE = 7


def weight(agent: Dict[str, Any]) -> float:
    """Return the selection weight for a curator agent."""
    return agent["stake"] * (1 + BETA_REP_WEIGHT * max(0, agent["rep"]))


def assign_curators(params: Dict[str, Any], step: int, sL: List[Any], s: Dict[str, Any]) -> Dict[str, Any]:
    """Assign curators to pending reports and record hidden votes."""
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

# ===== File: initial_state.py =====

"""Global constants and initial state for the Skygate simulation."""

# ---------- CONSTANTS ----------
MAX_SUPPLY = 1_000_000_000  # total cap
EMISSION_PER_REPORT = 3_500  # new SKY minted when a report is resolved
REWARD_SPLIT = dict(
    curator=0.70,  # 70 % to correct curators
    witness=0.20,  # 20 % to witness
    treasury=0.10,  # 10 % to DAO treasury
)
SLASH_FRACTION = 0.20  # 20 % of stake lost by wrong curators
STAKE_PER_REPORT = 100  # SKY each curator locks on a case
WITNESS_DEPOSIT = 50  # SKY bond per submission
MIN_CURATOR_STAKE = 500
MIN_AUDITOR_STAKE = 2_000
BETA_REP_WEIGHT = 0.05  # β in weight formula
PRICE_ALPHA = 0.30  # market sensitivity parameter
GROWTH_RATE = 0.02  # 2 % monthly report growth

# ---------- INITIAL STATE (month 0) ----------
state = dict(
    t=0,
    price=0.10,  # start at $0.10
    S_total=10_000_000,  # initial minted supply
    S_treasury=1_000_000,
    curators=[dict(stake=MIN_CURATOR_STAKE, rep=0, risk="med") for _ in range(50)],
    auditors=[dict(stake=MIN_AUDITOR_STAKE, rep=5, risk="low") for _ in range(5)],
    reports_pending=[],
)

# ===== File: process_reports.py =====

"""Report processing logic for Skygate.

This module defines a helper function ``process_reports`` which is
responsible for applying report outcomes and distributing rewards.
The code is based on the snippet labelled "SKYGATE – CHUNK 4".

The constants below are simplistic stand-ins. In a full implementation
these would likely be provided by other modules or configuration.
"""

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
    """Process all pending reports and update global state.

    Parameters
    ----------
    params : dict
        Simulation parameters (unused here but kept for API compatibility).
    step : int
        Current timestep of the simulation.
    sL : dict
        Previous state (unused here but part of cadCAD's API).
    s : dict
        Mutable simulation state containing a ``reports_pending`` list along
        with token and staking information.

    Returns
    -------
    dict
        Updated state ``s`` after processing all pending reports.
    """

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


# ===== File: report_process.py =====

"""Exogenous process for report arrivals.

This module implements the monthly arrival of witness reports based on a
demand curve.
"""

from math import ceil
from typing import List, Dict, Any

# Default growth rate used in the demand curve R(t) = 100 * (1+g)^t
GROWTH_RATE: float = 0.05

# Default deposit required for each witness report
WITNESS_DEPOSIT: int = 10


def report_arrival(params: Dict[str, Any], step: int, sL: List[dict], s: dict) -> Dict[str, List[Dict[str, Any]]]:
    """Generate new witness reports for the current timestep.

    Parameters
    ----------
    params : dict
        Parameters provided by cadCAD, must include a random number generator
        under ``rng`` and a ``truth_ratio`` controlling the fraction of
        legitimate reports.
    step : int
        Current cadCAD step (unused).
    sL : list
        State history (unused).
    s : dict
        Current state dictionary. Expects ``t`` indicating the month index.

    Returns
    -------
    dict
        Dictionary with key ``"new_reports"`` mapping to the list of generated
        reports.
    """
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
    """Append newly arrived reports to the pending queue."""
    s["reports_pending"].extend(_input["new_reports"])
    return s

# ===== File: sim.py =====

# Refactored and commented version of the user's cadCAD simulation model

# ─────────────────────────────────────────────────────────────
# STEP 1: Imports
# ─────────────────────────────────────────────────────────────
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from cadCAD.configuration import Configuration
from cadCAD.configuration.utils import config_sim
from collections import deque
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import ceil

np.random.seed(0)


# Constant used in the supply and price recomputation step
PRICE_ALPHA = 0.05
=======
# Minimum stake required to remain an active curator
MIN_CURATOR_STAKE = 100


@dataclass
class Report:
    """Simple container for report metadata."""
    id: int
    quality: float
    emission: float
    validated: bool
    submitter_staked: bool
    curators: list
    auditors: list
    age: int = 0

# ─────────────────────────────────────────────────────────────
# STEP 2: Initial State
# ─────────────────────────────────────────────────────────────
initial_state = {
    'emission_pool': 800_000_000,  # Starting emission pool
    'token_balances': {  # Token balances per stakeholder type
        'submitters': 0,
        'curators': 0,
        'auditors': 0,
        'governance': 0
    },
    'active_reports': [],  # List of active reports
    'report_id_counter': 0,  # Unique report IDs
    'accrued_apr': [],  # Reports earning APR
    'token_price': 1.0  # Starting token price in arbitrary units
}

# ─────────────────────────────────────────────────────────────
# STEP 3: Policy Functions
# ─────────────────────────────────────────────────────────────
def submit_report_policy(params, step, sL, s):
    """Generate validated reports for this timestep."""
    reports = []
    for _ in range(params['reports_per_step']):
        quality_score = np.random.uniform(50, 100)
        emission = params['min_emission'] + (quality_score / 100) * (
            params['max_emission'] - params['min_emission']
        )
        report = Report(
            id=s['report_id_counter'] + len(reports),
            quality=quality_score,
            emission=emission,
            validated=True,
            submitter_staked=True,
            curators=[True] * params['curators_per_report'],
            auditors=[True] * params['auditors_per_report'],
        )
        reports.append(report)
    return {'new_reports': reports}

def apr_accrual_policy(params, step, sL, s):
    """Compute APR rewards for reports that are accruing."""
    apr_updates = [
        0.20 * report.emission / 100  # 1% of the 20% stake APR per timestep
        for report in s['accrued_apr']
    ]
    return {'apr_rewards': sum(apr_updates)}


def recompute_supply_and_price(params, step, sL, s):
    """Recompute circulating supply and adjust the price."""
    prev = sL[-1] if sL else None

    # locked tokens
    stake_locked = sum(c["stake"] for c in s.get("curators", [])) + \
                   sum(a["stake"] for a in s.get("auditors", []))
    deposits_locked = sum(rep["deposit"] for rep in s.get("reports_pending", []))
    gov_locked = 0  # ignore for v1

    S_locked = stake_locked + deposits_locked + gov_locked
    s["S_circ"] = s.get("S_total", 0) - s.get("S_treasury", 0) - S_locked

    # utility demand
    D_util = S_locked

    if prev:
        delta_d = D_util - prev.get("D_util", 0)
        delta_s = s["S_circ"] - prev.get("S_circ", 0)
        s["price"] = s.get("price", 0) * (
            1 + PRICE_ALPHA * ((delta_d - delta_s) / prev.get("S_circ", 1))
        )

    # store for next step’s delta calc
    s["D_util"] = D_util
    return s

# ─────────────────────────────────────────────────────────────
# STEP 4: State Update Functions
# ─────────────────────────────────────────────────────────────
def update_active_reports(params, step, sL, s, _input):
    """Append new reports to the active list."""
    return 'active_reports', s['active_reports'] + _input['new_reports']

def update_report_counter(params, step, sL, s, _input):
    """Increment the running report ID counter."""
    return 'report_id_counter', s['report_id_counter'] + len(_input['new_reports'])

def update_token_balances(params, step, sL, s, _input):
    """Distribute rewards to stakeholders based on emission."""
    token_balances = s['token_balances'].copy()
    for report in _input['new_reports']:
        emission = report.emission
        token_balances['submitters'] += params['submitter_reward_pct'] * emission
        token_balances['curators'] += params['curator_reward_pct'] * emission
        token_balances['auditors'] += params['auditor_reward_pct'] * emission
        token_balances['governance'] += params['governance_reward_pct'] * emission
    return 'token_balances', token_balances

def update_apr_rewards(params, step, sL, s, _input):
    """Apply APR rewards to submitters."""
    token_balances = s['token_balances'].copy()
    token_balances['submitters'] += _input['apr_rewards']
    return 'token_balances', token_balances

def accrue_apr_reports(params, step, sL, s, _input):
    """Track reports that continue accruing APR."""
    return 'accrued_apr', s['accrued_apr'] + _input['new_reports']

def update_emission_pool(params, step, sL, s, _input):
    """Decrease the emission pool without letting it go negative."""
    total_emission = sum(r.emission for r in _input['new_reports'])
    capped_emission = min(total_emission, s['emission_pool'])
    return 'emission_pool', s['emission_pool'] - capped_emission


def update_token_price(params, step, sL, s, _input):
    """Estimate token price based on remaining emission pool."""
    total_emission = sum(r.emission for r in _input['new_reports'])
    capped_emission = min(total_emission, s['emission_pool'])
    new_pool = s['emission_pool'] - capped_emission
    utilization = (params['initial_emission_pool'] - new_pool) / params['initial_emission_pool']
    price = params['base_token_price'] * (1 + utilization)
    return 'token_price', price


# ─────────────────────────────────────────────────────────────
# SKYGATE – CHUNK 6
# Removes bankrupt / low-rep curators, adds new ones,
# advances month counter t += 1.
# ─────────────────────────────────────────────────────────────
def curator_churn(params, step, sL, s):
    """Evict low-quality curators and add newcomers as needed."""
    # evict
    s["curators"] = [
        c for c in s["curators"]
        if c["stake"] >= MIN_CURATOR_STAKE and c["rep"] > -5
    ]

    # add newcomers if needed
    desired = ceil(len(s["reports_pending"]) * 7 / 10)  # heuristic
    while len(s["curators"]) < desired:
        s["curators"].append(dict(stake=MIN_CURATOR_STAKE, rep=0, risk="med"))

    # advance month counter
    s["t"] += 1
    return s


# ─────────────────────────────────────────────────────────────
# STEP 5: Partial State Update Blocks
# ─────────────────────────────────────────────────────────────
psubs = [
    {
        'policies': {
            'submit_report': submit_report_policy,
        },
        'variables': {
            'active_reports': update_active_reports,
            'report_id_counter': update_report_counter,
            'token_balances': update_token_balances,
            'emission_pool': update_emission_pool,
            'token_price': update_token_price,
            'accrued_apr': accrue_apr_reports,
            'reports_per_step': lambda p, step, sL, s, _input: ('reports_per_step', p['reports_per_step']),
        }
    },
    {
        'policies': {
            'apr_accrual': apr_accrual_policy
        },
        'variables': {
            'token_balances': update_apr_rewards
        }
    }
]

# ─────────────────────────────────────────────────────────────
# STEP 6: Simulation Configuration
# ─────────────────────────────────────────────────────────────
sim_config = config_sim({
    'T': range(100),  # 100 timesteps
    'N': 30,  # Monte Carlo runs per configuration
    'M': {
        'min_emission': [200],
        'max_emission': [3500],
        'curators_per_report': [3],
        'auditors_per_report': [1],
        'reports_per_step': [100, 300, 500, 1000, 2000, 3500],
        'submitter_reward_pct': [0.20],
        'curator_reward_pct': [0.50],
        'auditor_reward_pct': [0.20],
        'governance_reward_pct': [0.10],
        'initial_emission_pool': [initial_state['emission_pool']],
        'base_token_price': [1.0],
    }
})

# ─────────────────────────────────────────────────────────────
# STEP 7: Run Simulation
# ─────────────────────────────────────────────────────────────
def run_simulation():
    """Execute the cadCAD simulation and return a results DataFrame."""
    exec_context = ExecutionContext(context=ExecutionMode().single_mode)
    configurations = [
        Configuration(
            user_id='user_1',
            model_id=f'skygate_v1_rps_{sc["M"]["reports_per_step"]}',
            subset_id='base',
            subset_window=deque(maxlen=1),
            initial_state=initial_state,
            partial_state_update_blocks=psubs,
            sim_config=sc,
        )
        for sc in sim_config
    ]

    executor = Executor(exec_context, configurations)
    raw_result, _, _ = executor.execute()

    df = pd.DataFrame(raw_result)
    df['tokens_submitters'] = df['token_balances'].apply(lambda x: x['submitters'])
    df['tokens_curators'] = df['token_balances'].apply(lambda x: x['curators'])
    df['tokens_auditors'] = df['token_balances'].apply(lambda x: x['auditors'])
    df['tokens_governance'] = df['token_balances'].apply(lambda x: x['governance'])
    df['token_price'] = df['token_price']
    return df

def plot_emission_pool(df):
    """Plot the remaining emission pool over time."""
    plt.figure(figsize=(10, 6))
    for rps in sorted(df['reports_per_step'].dropna().unique()):
        group = df[df['reports_per_step'] == rps]
        avg_emission = group.groupby('timestep')['emission_pool'].mean()
        plt.plot(avg_emission.index, avg_emission.values, label=f'{int(rps)} reports/step')

    plt.title("Emission Pool Over Time (Monte Carlo Averaged)")
    plt.xlabel("Timestep")
    plt.ylabel("Remaining Emission Pool")
    plt.legend(title="Reports/Step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cumulative_reports(df):
    """Plot how many reports have been submitted over time."""
    plt.figure(figsize=(10, 6))
    for rps in sorted(df['reports_per_step'].dropna().unique()):
        group = df[df['reports_per_step'] == rps].copy()
        group['num_reports_this_step'] = group['active_reports'].apply(len)
        group['cumulative'] = group.groupby('run')['num_reports_this_step'].cumsum()
        avg_cumulative = group.groupby('timestep')['cumulative'].mean()
        plt.plot(avg_cumulative.index, avg_cumulative.values, label=f'{int(rps)} reports/step')

    plt.title("Cumulative Reports Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Average Cumulative Reports Submitted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_token_price(df):
    """Plot the predicted token price over time."""
    plt.figure(figsize=(10, 6))
    for rps in sorted(df['reports_per_step'].dropna().unique()):
        group = df[df['reports_per_step'] == rps]
        avg_price = group.groupby('timestep')['token_price'].mean()
        plt.plot(avg_price.index, avg_price.values, label=f'{int(rps)} reports/step')

    plt.title("Predicted Token Price Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Token Price")
    plt.legend(title="Reports/Step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = run_simulation()
    plot_emission_pool(df)
    plot_cumulative_reports(df)
    plot_token_price(df)

