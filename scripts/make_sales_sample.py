"""يولّد داتاسيت مبيعات تجريبي لمغسلة سيارات — 6 أشهر."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
start = pd.Timestamp("2026-05-01")
days = 184

services = [
    ("Exterior Wash", 45, 12),
    ("Interior Steam Clean", 70, 20),
    ("Full Detailing", 150, 55),
    ("Silver Shine Package", 150, 52),
    ("Wax Gift + Wash", 100, 35),
    ("Engine Bay Clean", 60, 18),
    ("Tire & Rim Care", 30, 8),
    ("Ceramic Coating", 400, 160),
]
clients = [
    "Sari Refai", "Ahmad Omar", "Lina Haddad", "Yousef Nasser", "Rana Khalil",
    "Omar Zaid", "Dana Sami", "Khaled Mansour", "Maya Faris", "Tariq Aziz",
    "Fleet - Al Salam Co", "Fleet - Jordan Taxi", "Fleet - Aramex",
]
statuses = ["Completed"] * 90 + ["Pending"] * 7 + ["Cancelled"] * 3

rows = []
for d in range(days):
    date = start + pd.Timedelta(days=d)
    weekday = date.dayofweek
    base = 3.2 if weekday in (4, 5) else 2.2
    month_growth = 1 + d / days * 0.45
    n = max(1, int(rng.poisson(base * month_growth)))

    if date == start + pd.Timedelta(days=95):
        n = 18

    for _ in range(n):
        name, price, cost = services[rng.integers(0, len(services))]
        amount = float(price)
        if rng.random() < 0.15:
            amount = round(amount * rng.uniform(0.8, 0.95), 2)
        rows.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Client_Info": clients[rng.integers(0, len(clients))],
            "Service_Type": name,
            "Quantity": 1,
            "Amount": amount,
            "Cost": round(cost * rng.uniform(0.9, 1.1), 2),
            "Status": statuses[rng.integers(0, len(statuses))],
        })

df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
out = Path(__file__).resolve().parent.parent / "data" / "raw" / "sales_sample.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, encoding="utf-8-sig")
print("rows:", len(df), "| days:", df["Date"].nunique())
print("revenue:", round(df["Amount"].sum(), 2), "| cost:", round(df["Cost"].sum(), 2))
