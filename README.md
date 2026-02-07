
# 📈 Stock Market Volatility Regime Prediction Using Financial Networks

## Overview

This project investigates whether **financial network features** extracted from stock return correlations can improve the prediction of **market volatility regimes** compared to traditional return-based features.

We construct **correlation-based stock networks**—specifically the **Minimum Spanning Tree (MST)** and **Planar Maximally Filtered Graph (PMFG)**—on rolling windows of stock returns and extract structural network metrics. These features are then used to predict whether the market is in a **high-volatility** or **low-volatility** regime over a future horizon.

---

## Key Ideas

* Stock correlations encode collective market behavior
* Network topology captures structural changes during stress periods
* Network-derived features may be more informative than raw returns
* Rolling network analysis allows time-localized prediction

---

## Methodology

### 1. Data

* Universe: **NIFTY 50 stocks**
* Data source: Yahoo Finance
* Input: Daily log returns
* Time period: Post-2010 (pre-2010 market structure considered unstable)

---

### 2. Network Construction

For each rolling window of **60 trading days**, we compute the correlation matrix and transform it into a distance matrix using the Mantegna distance:

[
d_{ij} = \sqrt{2(1 - C_{ij})}
]

From this distance matrix, we construct:

* **Minimum Spanning Tree (MST)**
* **Planar Maximally Filtered Graph (PMFG)**

These networks preserve the strongest dependencies while filtering noise.

---

### 3. Feature Engineering

From each network, we extract structural features such as:

* Average shortest path length
* Network efficiency
* Diameter
* Maximum degree
* Degree entropy
* Betweenness centrality statistics
* Modularity (for PMFG)

We evaluate four feature sets:

1. **Baseline**: Log returns only
2. **Network features** (MST / PMFG)
3. **PCA-reduced network features** (5 components)
4. **Combined comparisons**

---

### 4. Prediction Task

* **Task**: Binary classification
* **Target**: Volatility regime (High vs Low)
* **Prediction horizon**: Next 60 trading days
* **Labeling**: Based on future realized volatility

---

### 5. Models Used

* **Logistic Regression (scikit-learn)**
* **PyTorch-based unsupervised regression model**

Models are evaluated using:

* Accuracy
* ROC-AUC

---

## Results

| Feature Set            | ROC-AUC   | Accuracy  |
| ---------------------- | --------- | --------- |
| Log returns (baseline) | ~0.59     | ~0.58     |
| Network features       | ~0.63     | ~0.54**   |
| PCA (5 components)     | ~0.64     | ~0.57     |

**Key takeaway:**
Network-based features significantly outperform raw return-based features, even with large rolling windows. The network based features is more conservative in predicting high volitality, wheras the log returns based features mostlypredicts high volitality

---

## Repository Structure

```text
STOCK_NETWORK/
│
├── datas/                  # Raw and processed datasets
├── figures/                # Network visualizations
├── src/                     # Core source code
│   ├── data_extraction.py   # Data ingestion & preprocessing
│   ├── metrics.py           # Network & classification metrics
│   ├── models.py            # ML & deep learning models
│   ├── plotting.py          # Visualization utilities
│   ├── utils.py             # Helper functions
│   └── __init__.py
│
├── network_analysis.ipynb   # Main analysis notebook
├── data_cleaning.ipynb     # Data preprocessing
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the analysis

Open and run:

```text
network_analysis.ipynb
```

All experiments and plots are reproducible from this notebook.

---

## Visualizations

The `figures/` directory contains example PMFG visualizations for different years, illustrating how market structure evolves over time.

---

## Notes & Limitations

* Rolling window size is fixed at 60 days
* Regime labels depend on volatility threshold choice
* Network construction is computationally expensive
* Results are indicative, not predictive trading signals

---


## Final Remarks

This project demonstrates that **financial network topology contains predictive information** about future market volatility regimes. Even simple classifiers benefit from network-derived features, highlighting the value of structural market analysis.

---
