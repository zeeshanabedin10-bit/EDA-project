# 🏠 House Price EDA & Prediction Project

> **A complete data science project** — from raw data exploration to trained machine learning models that predict house sale prices.

---

## 📁 Project Structure

```
eda project/
│
├── 📓 EDA_House_Prices.ipynb            ← Step 1: Exploratory Data Analysis
├── 📓 02_Feature_Engineering.ipynb      ← Step 2: Feature Engineering
├── 📓 03_Machine_Learning.ipynb         ← Step 3: ML Models & Evaluation
│
├── 📊 train.csv                         ← Ames Housing Dataset (1,460 rows × 81 cols)
├── 📊 train_engineered.csv              ← Model-ready dataset (after engineering)
│
├── 🖼️ eda_dashboard.png                 ← Final EDA summary dashboard
├── 🖼️ plot_01_saleprice_distribution.png
├── 🖼️ plot_02_price_by_quality.png
├── 🖼️ plot_03_price_by_neighborhood.png
├── 🖼️ plot_04_countplots.png
├── 🖼️ plot_05_scatter_plots.png
├── 🖼️ plot_06_pairplot.png
├── 🖼️ plot_07_heatmap.png
├── 🖼️ plot_08_outliers.png
├── 🖼️ plot_09_engineered_features.png
├── 🖼️ plot_10_log_transform.png
├── 🖼️ plot_11_model_comparison.png
├── 🖼️ plot_12_actual_vs_predicted.png
├── 🖼️ plot_13_residuals.png
├── 🖼️ plot_14_feature_importance.png
└── 🖼️ plot_15_lasso_coefficients.png
```

---

## 🎯 Project Objective

Use the **Ames Housing Dataset** (Kaggle) to:
1. Understand what factors drive house sale prices through EDA
2. Engineer richer features from domain knowledge
3. Build and compare regression models to predict price

**No machine learning was used in Step 1** — pure EDA as specified.

---

## 🔍 Step 1: Exploratory Data Analysis

### 6 Guiding Questions & Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Is SalePrice skewed? | ✅ Yes — right-skewed (skew ≈ 1.5). Log-transform normalises it. |
| 2 | Strongest correlates? | ✅ `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF` |
| 3 | Does OverallQual matter? | ✅ Yes — price rises monotonically from rating 1→10 |
| 4 | Neighborhood differences? | ✅ Top neighborhoods have 2× median price of bottom ones |
| 5 | Outliers? | ✅ A few very large homes with anomalously low price (non-standard sales) |
| 6 | Missing = unknown? | ✅ No — most NaN means "feature absent" (no pool, no alley) |

### Key EDA Plots

| Plot | Insight |
|------|---------|
| SalePrice Distribution | Right-skewed; log-transform makes it normal |
| Price by OverallQual | Strong, ordered relationship: quality = price |
| Price by Neighborhood | NridgHt, NoRidge, StoneBr command premium prices |
| Correlation Heatmap | GarageCars ↔ GarageArea highly collinear |
| Outlier Scatter | 2 large-area, low-price anomalies identified |

---

## 🔧 Step 2: Feature Engineering

### New Features Created

| Feature | Formula | Why |
|---------|---------|-----|
| `TotalSF` | `GrLivArea + TotalBsmtSF` | Total finished area > either alone |
| `HouseAge` | `YrSold - YearBuilt` | Newer houses = higher price |
| `RemodAge` | `YrSold - YearRemodAdd` | Recent remodel adds value |
| `IsRemodeled` | `YearRemodAdd != YearBuilt` | Binary flag |
| `IsNew` | `YrSold == YearBuilt` | Brand-new homes command premium |
| `QualXArea` | `OverallQual × GrLivArea` | Quality–size interaction |
| `TotalBath` | `FullBath + 0.5×Half + Bsmt` | Weighted bathroom count |
| `TotalPorchSF` | Sum of all porch types | Combined outdoor space |
| `HasPool` / `HasGarage` / etc. | Binary flags | Presence vs absence |

### Encoding
- **Ordinal** columns (quality ratings) → integers 0–5
- **Categorical** columns → one-hot encoded
- **Target** `SalePrice` → `log(SalePrice)` for normality

---

## 🤖 Step 3: Machine Learning Results

### Models Trained

| Model | Description | Scaling Needed |
|-------|-------------|---------------|
| Linear Regression | Baseline, fully interpretable | Yes |
| Ridge (L2) | Handles correlated features | Yes |
| Lasso (L1) | Auto feature selection | Yes |
| Gradient Boosting | Sequential trees, high accuracy | No |
| Random Forest | Ensemble of independent trees | No |

### Performance (on held-out test set)

> *Metrics are on log-price scale. Lower RMSE = better. Higher R² = better.*

| Model | Test RMSE | Test R² | CV RMSE |
|-------|-----------|---------|---------|
| Lasso (`alpha=0.0005`) | 0.1213 | 0.9127 | 0.1128 ± 0.0091 |
| Ridge (`alpha=10`) | 0.1230 | 0.9102 | 0.1136 ± 0.0089 |
| Tuned Gradient Boosting | 0.1242 | 0.9085 | — |
| Gradient Boosting | 0.1261 | 0.9056 | 0.1172 ± 0.0120 |
| Linear Regression | 0.1279 | 0.9030 | 0.1255 ± 0.0134 |
| Random Forest | 0.1442 | 0.8767 | 0.1357 ± 0.0090 |

> **Best model (Tuned Lasso):** Typical prediction error (MAE) = **\$14,103 (7.8% of mean price \$181,369)**; Dollar RMSE = **\$19,699**. Saved to `best_house_price_model.joblib`.

### Top Features (from tree models)
1. `OverallQual` — single most powerful predictor
2. `TotalSF` / `GrLivArea` — size matters
3. `QualXArea` — quality × size interaction
4. `Neighborhood` (encoded) — location premium
5. `HouseAge` — newer = more expensive

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Pandas | Data loading, manipulation |
| NumPy | Numerical operations |
| Matplotlib | Custom plots |
| Seaborn | Statistical visualisations |
| SciPy | Statistical tests (skewness, z-score) |
| Scikit-learn | ML models, cross-validation, metrics |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip3 install numpy pandas matplotlib seaborn scipy scikit-learn jupyter

# 2. Open Jupyter
cd ~/Desktop/eda\ project
jupyter notebook

# 3. Run notebooks in order:
#    EDA_House_Prices.ipynb       ← EDA
#    02_Feature_Engineering.ipynb ← Feature engineering
#    03_Machine_Learning.ipynb    ← Models
```

---

## 📌 Key Learnings

1. **EDA first, always** — the heatmap revealed which features mattered before training a single model
2. **Missing ≠ unknown** — read the data dictionary before imputing
3. **Log-transform skewed targets** — dramatically improves linear model performance
4. **Feature engineering beats model complexity** — `TotalSF` outperformed many raw features
5. **Tree models win on tabular data** — Gradient Boosting and Random Forest outperform linear models here

---

## ⚠️ Limitations

- Real Ames Housing `train.csv` (1,460 rows × 81 columns) is used; a held-out 20% test split evaluates generalization, while an external Kaggle `test.csv` submission would require matching the same preprocessing pipeline
- Hyperparameter tuning used a focused grid for `Ridge`, `Lasso`, and `GradientBoostingRegressor`; broader tuning or XGBoost/LightGBM could improve results further

---

*Project completed as part of an EDA course assignment.*
