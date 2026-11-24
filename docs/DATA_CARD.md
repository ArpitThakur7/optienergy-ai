# Data card

**Source:** Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54. **License:** CC BY 4.0. **Scope:** one household in Sceaux, France, minute records from 2006 to 2010.

**Observed processing counts:** 2,075,259 minute records, 25,979 invalid/missing minutes, 34,085 complete observed hours. Raw text SHA-256: `4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7`.

**Units:** minute-average power in kW, aggregated as sum(power)/60 kWh for a complete hour. The index denotes the right endpoint. Targets are never imputed; short input-only forward fill is bounded at six hours. Calendar timestamps are kept as naive local-clock values; no unsupported DST reconstruction is applied.

**Common-support evaluation coverage:** 33,174 / 33,885 causal supervised rows (97.90%). Weekly/daily seasonal reference availability removes some additional rows. This restricts the evaluated population of hours; it is not an all-hours accuracy claim.

**Privacy and generalization:** public historical measurements. The project does not identify occupants, infer protected characteristics, or establish representativeness of other households. No raw minute archive, fitted pickle models, or credentials are committed to the GitHub folder.
