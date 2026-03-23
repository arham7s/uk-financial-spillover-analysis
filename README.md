# Modeling Interdependencies Between UK Financial Markets

A quantitative analysis of return dynamics and cross-sector spillovers across nine UK-listed financial stocks spanning 2000–2024, using classical econometric models, deep learning, and a novel VAR-LSTM hybrid architecture.

---

## Overview

This project investigates how returns co-move and transmit shocks across three segments of the UK financial sector — **banks**, **insurers**, and **real estate investment trusts (REITs)** — using a multi-model forecasting and causality framework. The analysis covers over two decades of daily price data and is structured around five research questions:

1. How accurately can log returns be forecast by VAR, ARIMA, LSTM, and a hybrid model?
2. Do Granger-causal links flow more within sectors or across them?
3. Are there long-run equilibrium relationships (cointegration) among UK financial stocks?
4. Did structural shocks (Brexit, COVID, Mini-Budget) alter model parameters and causality density?
5. Do forecast improvements translate into economic value under a simple trading strategy?

---

## Dataset

| Field | Details |
|---|---|
| **Securities** | BARC, HSBA, LLOY, NWG, STAN (Banks) · AV, LGEN, PRU (Insurers) · DLN, LAND (Real Estate) |
| **Frequency** | Daily (business days) |
| **Period** | January 2000 – November 2024 (~2,493 observations) |
| **Source** | CSV file (`dataset.csv`) with adjusted closing prices |
| **Target variable** | Log returns: `100 × (ln Pₜ − ln Pₜ₋₁)` |

> HSBA was retained at the price-level stage but dropped before return modelling after passing the ADF stationarity test at the 5% level. All other nine series required first differencing (log returns) to achieve stationarity.

**Train/test split:** 90% training (≈ 2,242 obs) / 10% test (250 obs, corresponding to approximately July 2022 – November 2024).

---

## Methods

### 1. Stationarity Testing — Augmented Dickey-Fuller (ADF)
All price series are tested for unit roots. Non-stationary series are transformed to log returns. Stationarity of log return series is confirmed before model fitting.

### 2. Vector Autoregression (VAR)
A multivariate VAR(1) is fit on the nine log return series jointly, capturing linear cross-stock lead-lag dynamics. Residuals are tested with the Ljung-Box test to assess whether unexplained structure remains.

### 3. ARIMA
Individual ARIMA(1,0,1) models are fit per series as a univariate linear baseline.

### 4. LSTM (Long Short-Term Memory)
A univariate LSTM is trained per series with dropout regularisation and early stopping. Architecture: `LSTM(50) → Dropout(0.2) → Dense(1)`, trained with a 5-step lookback window.

### 5. VAR-LSTM Hybrid
A two-stage model:
- **Stage 1:** Fit VAR on training data; extract in-sample residuals.
- **Stage 2:** Train LSTM on residuals where the Ljung-Box test confirms autocorrelation structure remains (all 9 series pass).
- **Forecast:** VAR prediction + LSTM residual correction.

### 6. Johansen Cointegration Test & VECM
The Johansen trace and max-eigenvalue tests identify 2 cointegrating vectors among the log-price series at the 5% level, motivating a Vector Error Correction Model (VECM). Error correction coefficients (ECT loadings) are small across all stocks, indicating weak adjustment to the long-run equilibrium.

### 7. Diebold-Mariano (DM) Test
Statistical significance of forecast accuracy differences between models is assessed using squared-error loss. Benchmark: ARIMA.

### 8. Conditional Direction Accuracy
Directional accuracy (DA) is evaluated conditional on absolute returns exceeding thresholds τ ∈ {0.25%, 0.5%, 1.0%} to filter out near-zero predictions that inflate unconditional DA for LSTM.

### 9. Granger Causality & Sector Spillover Network
Pairwise Granger causality tests (max lag = 5, α = 0.05) produce a 9×9 binary causality matrix. Results are aggregated by sector to compute intra- and inter-sector spillover densities.

### 10. Structural Break & Regime Analysis (Chow Test)
VAR parameter stability is tested around three macro events:
- **Brexit vote** — 23 June 2016
- **COVID crash** — 16 March 2020
- **UK Mini-Budget** — 23 September 2022

### 11. Trading Strategy Backtest
Each model's directional forecasts drive a long/flat strategy (go long if predicted return > 0, otherwise hold cash). Performance is evaluated via annualised Sharpe ratio with 5 bps transaction costs.

---

## Key Results

### Forecast Accuracy (RMSE on test set)

| Stock | VAR | ARIMA | LSTM | Hybrid |
|---|---|---|---|---|
| BARC | 3.082 | 3.086 | 3.100 | 4.287 |
| LLOY | 2.335 | 2.344 | 2.333 | 3.587 |
| NWG | 3.424 | 3.425 | 3.492 | 3.815 |
| STAN | 3.338 | 3.338 | 3.370 | 3.431 |
| AV | 2.616 | 2.616 | 2.618 | 2.776 |
| LGEN | 2.605 | 2.605 | 2.603 | 2.748 |
| PRU | 3.041 | 3.040 | 3.039 | 3.672 |
| DLN | 2.844 | 2.849 | 2.866 | 3.148 |
| LAND | 2.931 | 2.931 | 2.984 | 3.116 |

> VAR and ARIMA perform near-identically across all stocks. The corrected LSTM (lookback=5) matches but does not significantly beat these baselines. The hybrid does not improve on either component — the VAR already captures the exploitable linear structure.

### Diebold-Mariano Test (vs ARIMA baseline)

| Stock | DM stat (LSTM) | p-value |
|---|---|---|
| BARC | 1.796 | 0.074 * |
| NWG | 2.220 | 0.027 ** |
| STAN | 1.690 | 0.092 * |
| LLOY, AV, LGEN, PRU, DLN, LAND | — | n.s. |

Statistically significant LSTM improvements are limited to 3 of 9 stocks at the 10% level.

### Cointegration

Johansen tests confirm **2 cointegrating vectors** at the 5% level. However, ECT loadings in the VECM are small (−0.017 to +0.006), indicating that while a long-run equilibrium exists, individual stocks adjust very slowly to deviations.

### Sector Spillover Density

| Direction | Density |
|---|---|
| Banks → Banks (intra) | 0.583 |
| Insurers → Insurers (intra) | 0.500 |
| Real Estate → Real Estate (intra) | 0.500 |
| Banks → Insurers | 0.833 |
| Banks → Real Estate | 0.750 |
| Real Estate → Insurers | **1.000** |
| Average intra-sector | 0.528 |
| Average inter-sector | **0.750** |

**Finding:** Inter-sector spillover density (0.75) significantly exceeds intra-sector density (0.53), indicating that return shocks transmit freely across sector boundaries. Systemic risk in UK financial equities is not sector-contained.

### Structural Breaks (Chow Test)

| Event | Date | F-stat | p-value | Verdict |
|---|---|---|---|---|
| Brexit | 2016-06-23 | 1.193 | 0.302 | No break |
| COVID | 2020-03-16 | 2.064 | **0.035** | Structural break ✓ |
| Mini-Budget | 2022-09-23 | 0.665 | 0.739 | No break |

Granger causality density collapsed by **88.7%** post-COVID (0.611 → 0.069), consistent with a liquidity shock decoupling normal lead-lag dynamics.

### Trading Strategy Backtest (Annualised Sharpe Ratio)

| Stock | Buy & Hold | VAR | ARIMA | LSTM | VECM |
|---|---|---|---|---|---|
| BARC | 1.027 | 0.000 | 0.000 | 0.711 | **1.024** |
| STAN | 1.000 | 0.000 | 0.000 | **0.571** | 0.528 |
| PRU | −1.035 | −1.139 | −1.036 | **−0.373** | −1.036 |
| LAND | −0.418 | −0.332 | −0.332 | **0.035** | 0.000 |
| LGEN | −0.467 | −0.637 | −0.469 | −1.536 | **−0.350** |

No model consistently outperforms buy-and-hold across all stocks. VECM performs best on banking stocks; LSTM shows selective advantage on insurance and real estate.

---

## Repository Structure

```
├── Modeling_Interdependencies_Between_Commodities_and_UK_Financial_Markets.ipynb
├── dataset.csv                  # Input data (daily adjusted closing prices)
└── README.md
```

---

## Requirements

```
python >= 3.10
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
tensorflow >= 2.x
keras
pmdarima
networkx
scipy
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn tensorflow pmdarima networkx scipy
```

---

## Usage

1. Clone the repository and place `dataset.csv` in the working directory (update `data_path` in Cell 0 if needed).
2. Open the notebook in Jupyter or Google Colab.
3. Run all cells sequentially — each section is self-contained and labelled (Sections 1–21).

> **Note:** The notebook was developed and tested on Google Colab with a GPU runtime. LSTM training cells will be significantly slower on CPU.

---

## Sections at a Glance

| Section | Content |
|---|---|
| 1–2 | Data loading, preprocessing, descriptive statistics, correlation heatmap |
| 3–4 | ADF stationarity tests, log return transformation |
| 5 | Train/test split |
| 6 | VAR(1) model — fit, forecast, Ljung-Box residual diagnostics |
| 7 | LSTM — per-stock training, RMSE results |
| 8–9 | ARIMA(1,0,1) — per-stock fitting and forecasting |
| 10–12 | Combined forecast visualisation, RMSE comparison table |
| 13 | Granger causality pairwise tests, causality network graph |
| 14 | *(Reserved / exploratory)* |
| 15–16 | Diebold-Mariano test, unconditional and conditional direction accuracy |
| 17 | Johansen cointegration test, VECM estimation, ECT interpretation |
| 18 | VAR-LSTM hybrid model, diagnostic check, DM comparison |
| 19 | Sector-based spillover analysis, intra/inter density, network visualisation |
| 20 | Structural break analysis — Chow tests, Granger density shifts across regimes |
| 21 | Trading strategy backtest — long/flat rule, Sharpe ratio, return tables |

---

## Limitations

- ARIMA parameters are fixed at (1,0,1) across all stocks rather than optimised per series.
- The LSTM uses a simple univariate architecture; multivariate or attention-based extensions could improve cross-stock signal capture.
- Transaction costs are set uniformly at 5 bps; results may vary significantly under different cost assumptions or with short-selling enabled.

---

## License

This project is released for academic and educational purposes. No warranty is provided regarding the accuracy or fitness for any financial application.
