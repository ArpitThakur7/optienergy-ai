# Dataset cache and provenance

The included `processed/hourly.csv` contains the observed hourly aggregation of UCI Individual Household Electric Power Consumption (dataset 235). `processed/metadata.json` records the source-text SHA-256, completeness counts, and timestamp convention. This is actual observed UCI data, not synthetic demo data.

At index t, `energy_kwh` represents `[t-1 hour, t)` and is missing unless all 60 minute-average kW readings are valid. `observed_minutes` records the number of valid minute readings. No imputation is applied to this cache. Imputation of input features, when needed, happens causally downstream and never creates target labels.

Attribution: Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54. Licensed CC BY 4.0. The hourly aggregation is a transformation of that dataset.

The raw text and ZIP are excluded from the downloadable project archive to keep it small. Run `python -m src.data_loader --force` to download and reconstruct the cache. The code does not require the raw archive for normal dashboard use.
