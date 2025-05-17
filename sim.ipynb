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
    'accrued_apr': []  # Reports earning APR
}

# ─────────────────────────────────────────────────────────────
# STEP 3: Policy Functions
# ─────────────────────────────────────────────────────────────
def submit_report_policy(params, step, sL, s):
    reports = []
    for _ in range(params['reports_per_step']):
        quality_score = np.random.uniform(50, 100)
        emission = params['min_emission'] + (quality_score / 100) * (params['max_emission'] - params['min_emission'])
        report = {
            'id': s['report_id_counter'] + len(reports),
            'quality': quality_score,
            'emission': emission,
            'validated': True,
            'submitter_staked': True,
            'curators': [True] * params['curators_per_report'],
            'auditors': [True] * params['auditors_per_report'],
            'age': 0
        }
        reports.append(report)
    return {'new_reports': reports}

def apr_accrual_policy(params, step, sL, s):
    apr_updates = [
        0.20 * report['emission'] / 100  # 1% of the 20% stake APR per timestep
        for report in s['accrued_apr']
    ]
    return {'apr_rewards': sum(apr_updates)}

# ─────────────────────────────────────────────────────────────
# STEP 4: State Update Functions
# ─────────────────────────────────────────────────────────────
def update_active_reports(params, step, sL, s, _input):
    return 'active_reports', s['active_reports'] + _input['new_reports']

def update_report_counter(params, step, sL, s, _input):
    return 'report_id_counter', s['report_id_counter'] + len(_input['new_reports'])

def update_token_balances(params, step, sL, s, _input):
    token_balances = s['token_balances'].copy()
    for report in _input['new_reports']:
        emission = report['emission']
        token_balances['submitters'] += 0.20 * emission
        token_balances['curators'] += 0.50 * emission
        token_balances['auditors'] += 0.20 * emission
        token_balances['governance'] += 0.10 * emission
    return 'token_balances', token_balances

def update_emission_pool(params, step, sL, s, _input):
    total_used = sum(r['emission'] for r in _input['new_reports'])
    return 'emission_pool', s['emission_pool'] - total_used

def update_apr_rewards(params, step, sL, s, _input):
    token_balances = s['token_balances'].copy()
    token_balances['submitters'] += _input['apr_rewards']
    return 'token_balances', token_balances

def accrue_apr_reports(params, step, sL, s, _input):
    return 'accrued_apr', s['accrued_apr'] + _input['new_reports']

def update_emission_pool(params, step, sL, s, _input):
    total_emission = sum(r['emission'] for r in _input['new_reports'])
    # Cap if pool isn't large enough
    capped_emission = min(total_emission, s['emission_pool'])
    return 'emission_pool', s['emission_pool'] - capped_emission


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
    'T': range(10),  # 100 timesteps
    'N': 30,  # Monte Carlo runs per configuration
    'M': {
        'submitter_stake': [100],
        'curator_stake': [100],
        'min_emission': [200],
        'max_emission': [3500],
        'curators_per_report': [3],
        'auditors_per_report': [1],
        'reports_per_step': [100, 300, 500, 1000, 2000, 3500]  # Parameter sweep
    }
})

# ─────────────────────────────────────────────────────────────
# STEP 7: Run Simulation
# ─────────────────────────────────────────────────────────────
exec_context = ExecutionContext(context=ExecutionMode().single_mode)
configurations = [
    Configuration(
        user_id='user_1',
        model_id=f'skygate_v1_rps_{sc["M"]["reports_per_step"]}',
        subset_id='base',
        subset_window=deque(maxlen=1),
        initial_state=initial_state,
        partial_state_update_blocks=psubs,
        sim_config=sc
    )
    for sc in sim_config
]

executor = Executor(exec_context, configurations)
raw_result, _, _ = executor.execute()

# ─────────────────────────────────────────────────────────────
# STEP 8: Data Processing
# ─────────────────────────────────────────────────────────────
df = pd.DataFrame(raw_result)

# Extract token balances into flat columns
df['tokens_submitters'] = df['token_balances'].apply(lambda x: x['submitters'])
df['tokens_curators'] = df['token_balances'].apply(lambda x: x['curators'])
df['tokens_auditors'] = df['token_balances'].apply(lambda x: x['auditors'])
df['tokens_governance'] = df['token_balances'].apply(lambda x: x['governance'])

# ─────────────────────────────────────────────────────────────
# STEP 9: Visualization - Emission Pool
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# STEP 10: Visualization - Cumulative Reports
# ─────────────────────────────────────────────────────────────
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
