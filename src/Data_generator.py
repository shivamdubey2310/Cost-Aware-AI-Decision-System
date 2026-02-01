"""
generate_synthetic_cost_aware_data.py

Generates a synthetic dataset for a Cost-Aware AI Decision System
(Finance & Banking).

Use case:
- Predict fraud probability
- Optimize investigation decisions under capacity constraints
- Minimize expected financial loss (not maximize accuracy)

Output:
- cost_aware_cases.csv
"""

import numpy as np
import pandas as pd
from faker import Faker
import random

# ---------------- CONFIG ----------------
NUM_CASES = 120_000
FRAUD_RATE = 0.08          # 8% fraud cases
SEED = 42
OUTPUT_PATH = "cost_aware_cases.csv"
# ----------------------------------------

fake = Faker()
np.random.seed(SEED)
random.seed(SEED)

cases = []

for i in range(NUM_CASES):
    is_fraud = np.random.rand() < FRAUD_RATE

    # Transaction / claim amount (fraud tends to be higher)
    amount = (
        np.random.lognormal(mean=9.5, sigma=0.7)
        if is_fraud
        else np.random.lognormal(mean=8.3, sigma=0.6)
    )

    # Behavioral risk signals
    tx_velocity = np.random.poisson(12 if is_fraud else 4)
    unusual_location = np.random.binomial(1, 0.35 if is_fraud else 0.05)
    device_change = np.random.binomial(1, 0.4 if is_fraud else 0.1)
    account_age_days = max(
        1,
        int(np.random.normal(180 if is_fraud else 900, 300))
    )

    # Financial costs
    fraud_loss = amount * np.random.uniform(0.7, 1.1) if is_fraud else 0
    investigation_cost = np.random.uniform(500, 2000)

    cases.append({
        "case_id": f"CASE_{1000000 + i}",
        "transaction_amount": round(amount, 2),
        "tx_velocity_24h": tx_velocity,
        "unusual_location_flag": unusual_location,
        "device_change_flag": device_change,
        "account_age_days": account_age_days,
        "investigation_cost": round(investigation_cost, 2),
        "fraud_loss_if_missed": round(fraud_loss, 2),
        "label_fraud": int(is_fraud)
    })

df = pd.DataFrame(cases)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset generated: {OUTPUT_PATH}")
print(df.head())