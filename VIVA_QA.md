# 🎤 Viva & Presentation Q&A Cheat Sheet

Quick answers to the most common questions about this House Prices EDA & ML project.

---

## 1. EDA & Statistics Questions

**Q: Why did you log-transform `SalePrice`?**  
`SalePrice` is right-skewed (mean > median, with a long tail of expensive homes). Taking `np.log1p(SalePrice)` makes the distribution close to normal, which satisfies linear regression assumptions and prevents expensive houses from dominating error metrics.

**Q: Why is the median better than the mean for `SalePrice`?**  
Because `SalePrice` is right-skewed, a few high-priced homes pull the mean upward. The median represents a typical house price more accurately.

**Q: Why did the IQR method flag more outliers than the Z-score method?**  
Z-scores assume a normal distribution (using mean and standard deviation, which are pulled up by extreme values in right-skewed data). IQR uses quartiles (Q1 and Q3), which are robust to skewness, so it flags more points in the long right tail.

**Q: Why did you replace `NaN` with `"None"` in `PoolQC`, `Alley`, `Fence`, `FireplaceQu`, and `GarageType` instead of dropping or imputing with mode?**  
In the Ames Housing data dictionary, `NaN` in these columns means the feature is **absent** (e.g., no pool or no garage), not that the value is missing/unknown. Replacing with `"None"` preserves real domain information.

**Q: Why did you impute `LotFrontage` using the median grouped by `Neighborhood`?**  
Lot frontage depends heavily on neighborhood layout and lot sizes. Imputing with the neighborhood median is more accurate than using a single global median.

**Q: Why is Pearson correlation more useful than covariance here?**  
Covariance depends on variable units (e.g., square feet × dollars produces huge, hard-to-compare numbers). Correlation normalizes covariance to `[-1, +1]`, making relationships directly comparable across features.

---

## 2. Feature Engineering Questions

**Q: Which engineered features were most valuable and why?**
- `TotalSF = GrLivArea + TotalBsmtSF` — buyers value total usable indoor space.
- `QualXArea = OverallQual * GrLivArea` — captures the interaction between size and build quality.
- `HouseAge = YrSold - YearBuilt` — newer houses generally sell for more.
- `TotalBath` — combines full and half baths above and below ground into one weighted count.

**Q: Why did you map `ExterQual` and `KitchenQual` to `0–5` instead of one-hot encoding?**  
They have a natural order (`None < Po < Fa < TA < Gd < Ex`). Ordinal integer encoding preserves that ranking with a single column instead of creating sparse dummy columns.

**Q: Why `drop_first=True` in `pd.get_dummies()`?**  
To avoid the dummy variable trap (perfect multicollinearity) in linear models.

---

## 3. Machine Learning Questions

**Q: Why did you scale features with `RobustScaler` for Linear/Ridge/Lasso, but not for Random Forest or Gradient Boosting?**  
Linear models with regularization penalize coefficient magnitudes, so features on different scales (e.g., `LotArea` vs `FullBath`) would be penalized unfairly without scaling. `RobustScaler` uses median and IQR, making it resistant to outliers. Decision trees split on feature thresholds, so they are scale-invariant.

**Q: Why did Lasso perform best on the real Ames dataset (`R² = 0.9127`, `RMSE = 0.1213`)?**  
After one-hot encoding, we had 217 features—many sparse or correlated (e.g., `GarageArea` vs `GarageCars`). Lasso's L1 penalty zeroed out 124/217 (57.1%) less-useful features, reducing overfitting and improving generalization.

**Q: What is the difference between Ridge (L2) and Lasso (L1)?**  
- **Ridge (L2):** shrinks coefficients smoothly toward zero, keeping all features; good for multicollinearity.
- **Lasso (L1):** shrinks unimportant coefficients to **exact zero**, performing automatic feature selection.

**Q: How do you interpret `RMSE = 0.1213` on log-price?**  
On a natural log scale, `0.1213` corresponds roughly to a **~12% typical relative error**. In actual dollars on the test set, the Tuned Lasso model achieved a **MAE of \$14,103** (7.8% of the \$181,369 mean price) and **Dollar RMSE of \$19,699**.
