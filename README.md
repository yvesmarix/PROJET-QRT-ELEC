# PROJET-QRT-ELEC — Electricity Price Modelling

> **QRT Data Challenge — "How to explain the price of electricity?"**

**Authors:** Nathan LIESSE & Yves-Marie SALIOU  
**Program:** Master 2 IEF — Université Paris-Dauphine (2025/2026)  
**Instructors:** Sylvain Benoit (Group 1), Arthur Thomas (Group 2)

---

## Overview

This project is part of the **QRT Data Challenge** hosted on the Collège de France platform ([challengedata.ens.fr](https://challengedata.ens.fr/challenges/97)). The goal is to **model the daily variations in electricity futures prices** (24h maturity) for **France** and **Germany**, using meteorological, energy, and commodity variables.

This is a **supervised regression** problem where the evaluation metric is the **Spearman correlation** between predictions and actual values — the focus is on correctly *ranking* observations rather than predicting exact values.

## Results

The best models significantly outperform the official benchmark (Spearman = **15.86%**):

| Model | Spearman CV (%) |
|---|---|
| Benchmark (linear regression, NaN → 0) | ~15.86 |
| Ridge (selected features) | ~20–21 |
| CatBoost optimized (Optuna) | ~22–23 |
| Ensemble Ridge + MLP on ranks | ~24 |
| **Kernel Ridge RBF per country + Optuna** | **~24–26** |

## Repository Structure

```
PROJET-QRT-ELEC/
├── LIESSE_SALIOU_QRT_Electricite.ipynb   # Main notebook (full pipeline)
├── X_train_NHkHMNU.csv                   # Training features (1,494 rows)
├── X_test_final.csv                       # Test features (654 rows)
├── y_train_ZAN5mwg.csv                    # Target variable (train)
├── y_test_random_final.csv                # Random submission example
├── submission_ensemble_3modeles.csv       # Final submission (3-model ensemble)
└── requirements.txt                       # Python dependencies
```

## Data

The dataset contains **35 columns** for 2 countries (FR, DE):

- **Identifiers**: `ID`, `DAY_ID`, `COUNTRY`
- **Commodity price returns** (daily): `GAS_RET`, `COAL_RET`, `CARBON_RET`
- **Weather by country**: `x_TEMP`, `x_RAIN`, `x_WIND`
- **Energy production by country**: `x_GAS`, `x_COAL`, `x_HYDRO`, `x_NUCLEAR`, `x_SOLAR`, `x_WINDPOW`, `x_LIGNITE`
- **Consumption & exchanges**: `x_CONSUMPTION`, `x_RESIDUAL_LOAD`, `x_NET_IMPORT`, `x_NET_EXPORT`, `DE_FR_EXCHANGE`, `FR_DE_EXCHANGE`

## Pipeline

The notebook follows a structured pipeline in 10 sections:

### 1. Data Preparation
- Exploratory analysis and descriptive statistics
- Train/test split verification (random sampling, not a time series split)

### 2. Processing & Feature Engineering
- **Imputation** of missing values using per-country median (computed on train only)
- **Feature engineering** guided by the economic theory of the *merit order*:
  - Effective marginal cost: `GAS_CARBON_COST`, `COAL_CARBON_COST`, `FOSSIL_COST_INDEX`
  - Residual demand and renewable penetration rate
  - Weather × production interactions: `WIND_x_WINDPOW`, `RAIN_x_HYDRO`, `TEMP_x_RESIDUAL`
  - Net cross-border flows
- **Normalization** via `QuantileTransformer` (Gaussian output), fitted on train only

### 3. Feature Selection
- **Permutation importance** on Random Forest (300 trees, 10 repeats)
- Selection threshold at 0.01 on Spearman score drop
- Cross-validation with **`GroupKFold`** on `DAY_ID` (FR/DE pairs from the same day always stay in the same fold)

### 4. Benchmark
- Reproduction of the official benchmark (linear regression, NaN → 0): Spearman ≈ 15.86%

### 5. Unsupervised Models
- **PCA** and **t-SNE** for visualization (clear FR/DE separation, no target clusters)
- **KMeans** and **GMM** tested as additional features → no significant improvement

### 6. Supervised Models
- **Linear**: Ridge, Lasso, ElasticNet (with CV for hyperparameters)
- **Boosting**: XGBoost, LightGBM, CatBoost (Bayesian optimization via Optuna)
- **Kernel Ridge RBF** trained separately per country (best model)
- **Per-country** vs joint modeling strategy

### 7. Model Interpretation
- Ridge coefficients, permutation importance, **SHAP** (beeswarm plots)
- All 3 methods converge: fossil costs (+), renewable production (−), residual demand (+)

### 8. Deep Learning
- **MLP** with Dropout + L2, and a Batch Normalization variant
- MLP on **target ranks** (better suited for the Spearman metric)
- Diagnosis: classic overfitting on a small dataset (1,494 rows)

### 9. Final Submission
- **Ensemble** of 3 models (Ridge + Kernel Ridge + MLP on ranks)

## Installation

```bash
git clone https://github.com/yvesmarix/PROJET-QRT-ELEC.git
cd PROJET-QRT-ELEC
pip install -r requirements.txt
```

### Dependencies

- `numpy`, `pandas`, `matplotlib`, `scipy`
- `scikit-learn`
- `xgboost`, `lightgbm`, `catboost`
- `optuna`
- (optional) `tensorflow`, `shap` — used in the notebook for the MLP and interpretation sections

## Usage

Open and run the notebook:

```bash
jupyter notebook LIESSE_SALIOU_QRT_Electricite.ipynb
```

The notebook is self-contained: it loads the CSVs from the current directory, runs the full pipeline, and generates the submission file `submission_ensemble_3modeles.csv`.

## Key Concepts

**Merit order** — The spot price of electricity is set by the marginal cost of the last power plant called to meet demand. Renewables (near-zero cost) lower the price by displacing expensive fossil fuel plants.

**Spearman correlation** — Measures rank concordance between predictions and targets. A model that correctly orders observations scores well even without predicting exact values.

**GroupKFold** — Cross-validation strategy ensuring that both observations from the same day (FR + DE) remain in the same fold, preventing information leakage.

## License

This project was completed in an academic context. Data comes from the QRT Data Challenge.
