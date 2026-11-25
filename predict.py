"""CLI & helper for predicting house prices using the trained Ames Housing model."""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_house_price_model.joblib"
DATA_PATH = BASE_DIR / "train_engineered.csv"


def load_artifacts():
    """Load saved model artifact and baseline feature medians."""
    artifact = joblib.load(MODEL_PATH)
    df_eng = pd.read_csv(DATA_PATH)
    medians = df_eng.drop(columns=["SalePrice"]).median().to_dict()
    return artifact, medians


def predict_price(
    overall_qual: int = 6,
    gr_liv_area: float = 1462.0,
    total_bsmt_sf: float = 991.0,
    garage_area: float = 480.0,
    year_built: int = 1973,
    full_bath: int = 2,
    neighborhood: str | None = None,
) -> float:
    """Predict house sale price in dollars given key property attributes."""
    artifact, medians = load_artifacts()
    feature_cols = artifact["feature_columns"]

    # Start from dataset medians for all 217 features
    row = pd.DataFrame([{col: medians.get(col, 0.0) for col in feature_cols}])

    # Override user-supplied inputs and recompute derived features
    row["OverallQual"] = overall_qual
    row["GrLivArea"] = gr_liv_area
    row["TotalBsmtSF"] = total_bsmt_sf
    row["GarageArea"] = garage_area
    row["YearBuilt"] = year_built
    row["FullBath"] = full_bath

    if "TotalSF" in row.columns:
        row["TotalSF"] = gr_liv_area + total_bsmt_sf
    if "QualXArea" in row.columns:
        row["QualXArea"] = overall_qual * gr_liv_area
    if "HouseAge" in row.columns and "YrSold" in row.columns:
        row["HouseAge"] = max(0, row["YrSold"].iloc[0] - year_built)
    if "HasGarage" in row.columns:
        row["HasGarage"] = int(garage_area > 0)
    if "HasBasement" in row.columns:
        row["HasBasement"] = int(total_bsmt_sf > 0)

    if neighborhood:
        for col in feature_cols:
            if col.startswith("Neighborhood_"):
                row[col] = 1 if col == f"Neighborhood_{neighborhood}" else 0

    X_input = row[feature_cols]
    if artifact["scaler"] is not None:
        X_input = artifact["scaler"].transform(X_input)

    log_pred = artifact["model"].predict(X_input)[0]
    return float(np.expm1(log_pred))


def main():
    parser = argparse.ArgumentParser(description="Predict Ames house sale price.")
    parser.add_argument("--qual", type=int, default=7, help="Overall quality (1-10)")
    parser.add_argument("--area", type=float, default=1800, help="Living area (sq ft)")
    parser.add_argument("--bsmt", type=float, default=1000, help="Basement area (sq ft)")
    parser.add_argument("--garage", type=float, default=500, help="Garage area (sq ft)")
    parser.add_argument("--year", type=int, default=2002, help="Year built")
    parser.add_argument("--bath", type=int, default=2, help="Full bathrooms")
    parser.add_argument("--neighborhood", type=str, default=None, help="Neighborhood code (e.g., NridgHt)")
    args = parser.parse_args()

    price = predict_price(
        overall_qual=args.qual,
        gr_liv_area=args.area,
        total_bsmt_sf=args.bsmt,
        garage_area=args.garage,
        year_built=args.year,
        full_bath=args.bath,
        neighborhood=args.neighborhood,
    )
    print(f"Estimated SalePrice: ${price:,.0f}")


if __name__ == "__main__":
    main()
