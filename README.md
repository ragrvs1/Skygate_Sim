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

The script will run multiple Monte Carlo simulations and generate several plots:

1. **Emission Pool Over Time** – average remaining tokens with one standard deviation error bands.
2. **Cumulative Reports Over Time** – average number of reports submitted.
3. **Circulating Supply Over Time** – tokens distributed to stakeholders.
4. **Cumulative Verified Reports Over Time** – running total of verified reports.
5. **Predicted Token Price Over Time** – simple price estimate based on emission pool utilization.
6. **Δ Utility Demand vs Δ Supply** – scatter showing price changes against net demand shifts.
7. **Price Sensitivity Analysis** – compare token price trajectories for multiple `PRICE_ALPHA` values.
8. **Active Curator Count** – number of curators participating over time.
9. **Stake Distribution** – box plot of curator stake each timestep.
10. **Slashing and Reward Flows** – minted rewards versus slashed stake.

## Project Structure

- `sim.py` – defines the simulation configuration, policies, and plotting utilities.
- `README.md` – project overview and usage instructions.

Feel free to modify parameters in `sim.py` (such as the number of reports per step) to explore different scenarios.
