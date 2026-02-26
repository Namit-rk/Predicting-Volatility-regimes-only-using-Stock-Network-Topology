# Predicting Volatility Regimes Using Stock-Network Topology

This repository studies whether **financial-network structure** (instead of only raw returns) helps predict future market volatility regimes.

The project builds stock-correlation networks from Indian equities, extracts graph-level metrics, and uses those features in binary classification models to predict whether the next horizon is likely to be **high-volatility** or **low-volatility**.

---

## Research motivation

Traditional volatility prediction models rely primarily on **time-series features** such as past returns, rolling variance, or GARCH-type estimators.
However, financial markets are not independent time series — they are **interacting systems**.

During market stress, assets stop behaving independently and begin moving collectively.
Empirically this appears as:

* correlations rising sharply during crashes
* sectoral boundaries weakening
* market-wide synchronization

Therefore, the central question of this project is:

> **Can structural information about how stocks are connected predict future volatility regimes better than individual return behavior alone?**

Instead of asking *“how volatile was the market?”*, we ask:

> **“How organized was the market?”**

Financial networks provide a way to quantify that organization.

---

## What this project does

* Builds rolling stock-correlation networks from log-return data
* Constructs two filtered graph representations:

  * **MST (Minimum Spanning Tree)**
  * **PMFG (Planar Maximally Filtered Graph)**
* Computes network descriptors (clustering, efficiency, modularity, degree entropy, centralization)
* Trains classifiers for volatility-regime prediction
* Compares network-feature performance against return-only baselines

---

## Repository layout

```
.
├── datas/
│   ├── raw_data.csv
│   ├── clean_data.csv
│   └── log_returns_data.csv
├── figures/
│   ├── pmfg_2008.png
│   ├── pmfg_2015.png
│   ├── pmfg_2020.png
│   └── pmfg_2008_2022.png
├── src/
│   ├── data_extraction.py
│   ├── utils.py
│   ├── metrics.py
│   ├── models.py
│   ├── plotting.py
│   └── __init__.py
├── data_cleaning.ipynb
├── network_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## End-to-end workflow

### 1. Collect universe and prices

* `get_top_indian_stocks_tickers()` scrapes NSE F&O symbols and maps to Yahoo Finance (`.NS`)
* `batch_download_price_data(...)` downloads close prices

### 2. Clean and transform data

* Missing data filtering and stock selection handled in `data_cleaning.ipynb`
* Log returns form the modeling input

### 3. Build network snapshots

* `make_PMFG_Network(df)` constructs planar filtered networks
* `make_MST_Network(df)` constructs the minimum spanning tree using correlation distance

### 4. Engineer network features

`compute_network_metrics(G)` extracts:

* clustering coefficient
* average shortest path
* efficiency
* modularity
* max degree
* degree entropy

### 5. Train classifiers

* `Regression(...)` — logistic regression (scikit-learn baseline)
* `Regression_Pytorch(...)` — logistic model implemented in PyTorch

### 6. Evaluate

`classification_metrics(...)` reports:

* Accuracy
* ROC-AUC
* Confusion matrix

---

## Why network filtering?

A raw correlation matrix is dense and extremely noisy.
For **N stocks**, there are `N(N−1)/2` pairwise relationships, most of which are spurious.

Filtered financial networks solve this.

### Minimum Spanning Tree (MST)

Keeps only the strongest dependencies needed to maintain connectivity (N−1 edges).
This extracts the market’s **dependency backbone**.

### Planar Maximally Filtered Graph (PMFG)

Allows more edges while preserving planarity.
This preserves sector clusters while still removing noise.

Intuition:

| Market State | Network Structure             |
| ------------ | ----------------------------- |
| Normal       | modular, sector-clustered     |
| Crisis       | centralized, highly connected |

Thus, **topology itself becomes a market state variable**.

---

## Volatility labeling

* Rolling windows: 60 trading days
* Prediction horizon: future 60 trading days
* Target: high vs low realized volatility

Realized volatility is computed from future returns and thresholded to define the regime.

---

## Data notes

* Wide format dataset (one column per ticker)
* First log-return row contains NaNs after differencing
* Sensitive to window size and labeling threshold

---

## Outputs

* PMFG visualizations in `figures/`
* Classification metrics
* Confusion matrices
* PCA analysis
* Network plots

---

## Key findings

The network features exhibited systematic structural changes across regimes.

### Low-volatility periods

* higher modularity
* sector clustering
* longer average path length
* decentralized structure

### High-volatility periods

* correlations spike
* network diameter shrinks
* central nodes emerge
* degree entropy decreases

**Interpretation**

Market crises are not merely periods of large price movement — they are periods of **synchronization**.

> Volatility increases when the market begins to behave like a single asset.

This suggests topology can be predictive because **structural organization changes before realized volatility spikes**.

---

## Setup

### Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```
.venv\Scripts\activate
```

### Install dependencies

```
pip install -r requirements.txt
```

---

## Running the project

Primary workflow is notebook-driven:

1. Run `data_cleaning.ipynb`
2. Run `network_analysis.ipynb`

---

## Quick start example

```python
import pandas as pd
from src.utils import make_PMFG_Network
from src.metrics import compute_network_metrics

returns = pd.read_csv(
    "datas/log_returns_data.csv",
    parse_dates=["date"],
    index_col="date"
).dropna()

G = make_PMFG_Network(returns)
features = compute_network_metrics(G)

print(features)
```

---

## Limitations and caveats

* Yahoo Finance ticker coverage may drift
* PMFG construction is computationally heavy
* Regime labels depend on volatility threshold
* Results are **research-oriented and not trading signals**

---

## What this project demonstrates

**Quantitative finance**

* volatility regime definition
* realized volatility forecasting

**Network science / econophysics**

* correlation distance transformation
* MST & PMFG construction
* topology interpretation

**Machine learning**

* feature engineering from graph structure
* binary classification with temporal dependence

**Software engineering**

* modular research code
* reproducible pipeline
* automated data ingestion

Central takeaway:

> Market volatility is not only a statistical phenomenon — it is a structural one, which could help in prediction of risk.

---

## Future work

Possible extensions:

* dynamic (time-evolving) financial networks
* graph neural networks on rolling graphs
* multi-asset networks (equities + bonds + commodities)
* systemic risk early-warning indicators


* your research thinking
* your math maturity
* and your market intuition.
