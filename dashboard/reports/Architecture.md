# Project architecture

```mermaid
flowchart TD
    A[Official UCI ZIP + SHA-256 provenance] --> B[Validate timestamps and minute power]
    B --> C[Hourly observed kWh + completeness]
    C --> D[Causal input fill and lag/rolling/calendar features]
    D --> E[Chronological 70/15/15 split + boundary purge]
    E --> F[Train-only feature and target transformations]
    F --> G[Fixed NumPy linear/ridge or PyTorch ANN]
    G --> H[Paired seeds and optimizer state initialization]
    H --> I[Training data-processing budget + monitoring counters]
    I --> J[Best validation checkpoints]
    J --> K[Mean validation configuration selection]
    K --> L[Final test metrics and residuals]
    K --> M[Trusted inference bundle]
    L --> N[CSV/JSON + Plotly/Matplotlib + PDF report]
    M --> O[Historical one-step / recursive forecasts]
    N --> P[Seven-page Streamlit dashboard]
    O --> P
    Q[Quadratic / Rosenbrock / Himmelblau] --> R[Shared NumPy optimizers + SciPy L-BFGS]
    R --> S[Contours / surfaces / gradient arrows / animation]
    S --> P
```

The training module never reads test arrays. The experiment runner owns final selection and test evaluation. Inference bundles contain the fitted transformations, model parameters, configuration, and training-period metadata; they do not embed the full dataset. The recursive prediction module truncates history at the origin before computing any features.
