# Energy Intelligence — One-Page Industrial Summary

**What we built.** A working prototype that turns your raw energy-meter data
into a live dashboard with reliability checks, day-ahead demand forecasts,
ranked anomaly alerts, and costed saving recommendations. It runs on open
building data today and plugs into your meter exports without redevelopment.

## Key insights from the reference analysis (8 buildings, 1 year)

- **Your data can be trusted — once checked.** Automated profiling found and
  repaired duplicate timestamps, sensor outages and impossible readings, and
  scores every meter 0–100 so degrading data quality is caught early.
- **Demand is predictable.** Day-ahead forecasts reach ~4 % average error —
  78 % better than the "same as yesterday" rule — enabling proactive load
  planning and peak avoidance.
- **Abnormal consumption is visible and ranked.** Night-time equipment left
  running, daytime outages and extreme spikes are detected automatically and
  sorted by severity, so your team reviews a short list, not raw data.
- **Concrete saving levers were quantified** (estimates at 0.25 €/kWh,
  120 €/kW·yr demand charge):
  - **Peak shaving:** a 10 % peak reduction is worth tens of k€/yr per
    building in demand charges; the dashboard shows exactly which hours to shift.
  - **Night/standby load:** buildings with > 25 % night consumption carry the
    largest kWh saving potential; a 30 % night cut is simulated per building.
  - **Anomaly follow-up:** each high-severity episode has an evidence trail
    ready for maintenance triage.

## Recommended actions

1. Review the top-5 dashboard recommendation cards with facility management.
2. Assign an owner to every high-severity anomaly episode (2-week cycle).
3. Pilot shutdown schedules in the building with the highest night share.
4. Provide 6–12 months of your own meter exports (timestamp, meter ID, kWh)
   to replace the open dataset — the pipeline swaps inputs, not methods.

## Next steps for a partnership

Data-sharing agreement → 4-week pilot on your data → validated saving
estimates with your real tariff → alerting integration (e-mail/Teams).

*All figures are estimates based on configurable tariff and CO₂ factors; the
methodology and code are fully documented and reproducible.*
