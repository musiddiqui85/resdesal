"""How long storage lasts and how much production must survive an outage (Section 3.1 of the paper).

    python examples/01_storage_and_surviving_production.py

Uses the closed-form relations of resdesal.screening for the GCRP-600 reference plant: one day of
production in storage (600,000 m³) and a critical demand of 40% of capacity (240,000 m³/d).
"""

from resdesal import GCRP600, bridge_days, required_sigma

P = GCRP600
b = bridge_days(P.storage, P.critical_demand)
print(f"Storage of {P.storage:,.0f} m³ bridges {b:.1f} days of critical demand ({P.critical_demand:,.0f} m³/d).\n")

print("Surviving production needed, as a share of critical demand (no start-up delay, no recovery):")
print(f"{'Outage':>8}  {'1 day of storage':>18}  {'7 days of storage':>18}")
for T in (3, 12, 55, 120):
    now = required_sigma(T, P.storage, P.critical_demand)
    seven = required_sigma(T, 7 * P.critical_demand, P.critical_demand)
    print(f"{T:>6} d  {now:>9.0%} ({now * P.critical_demand:>7,.0f})  {seven:>9.0%} ({seven * P.critical_demand:>7,.0f})")

# Backup that must be started uses storage while it starts.
for delay in (1.0, 2.0, 3.0):
    s = required_sigma(12.0, P.storage, P.critical_demand, startup_delay=delay)
    text = "storage runs out before the backup is online" if s == float("inf") else f"{s:.2f}"
    print(f"\n12-day outage with a {delay:g}-day start-up delay: {text}", end="")
print()

# Expected (paper, Section 3.1): 2.5 days; 79% for 12 days, 95% for 55 days, 98% for 120 days;
# 42% and 87% with 7 days of storage; 0.95 with a 2-day delay.
