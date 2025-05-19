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
