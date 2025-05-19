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

np.random.seed(0)

# Constant used in the supply and price recomputation step
PRICE_ALPHA = 0.05

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

