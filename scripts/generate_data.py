import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_CUSTOMERS = 10000
DAYS = 180

customers = [str(uuid.uuid4()) for _ in range(NUM_CUSTOMERS)]
cities = ["Mumbai", "Bangalore", "Delhi", "Pune", "Hyderabad"]
marketing_sources = ["organic", "paid", "referral", "promo"]

start_date = datetime.now() - timedelta(days=DAYS)
tx_per_day = np.random.randint(400, 800, size=DAYS)

rows = []
first_tx_date = {}

for d in range(DAYS):
    day = start_date + timedelta(days=d)
    n_tx = tx_per_day[d]

    for _ in range(n_tx):
        customer_id = np.random.choice(customers)

        if customer_id not in first_tx_date:
            first_tx_date[customer_id] = day

        date = day + timedelta(
            hours=int(np.random.uniform(9, 22)),
            minutes=int(np.random.uniform(0, 60))
        )

        revenue = float(np.clip(np.random.normal(250, 100), 50, 2000))
        cogs = revenue * np.random.uniform(0.6, 0.8)
        gross_margin = revenue - cogs

        marketing_source = np.random.choice(marketing_sources)

        if marketing_source == "organic":
            marketing_cost = 0.0
        elif marketing_source == "paid":
            marketing_cost = float(np.random.uniform(50, 200))
        elif marketing_source == "referral":
            marketing_cost = float(np.random.uniform(20, 80))
        else:
            marketing_cost = float(np.random.uniform(10, 60))

        opex_allocated = float(np.random.uniform(10, 40))
        items_count = int(np.random.randint(1, 10))
        city = np.random.choice(cities)
        promo_used = bool(np.random.choice([0, 1], p=[0.7, 0.3]))

        cohort_week = first_tx_date[customer_id].isocalendar().week

        rows.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id": customer_id,
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "revenue": round(revenue, 2),
            "cogs": round(cogs, 2),
            "gross_margin": round(gross_margin, 2),
            "marketing_source": marketing_source,
            "marketing_cost": round(marketing_cost, 2),
            "opex_allocated": round(opex_allocated, 2),
            "items_count": items_count,
            "city": city,
            "promo_used": promo_used,
            "customer_cohort": cohort_week
        })

df = pd.DataFrame(rows)
df.to_csv("data/transactions.csv", index=False)
print("Generated", len(df), "rows into data/transactions.csv")

