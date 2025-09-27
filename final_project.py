#!/usr/bin/env python3
"""
final_project.py
E-commerce Sales Data Analysis - single-file deliverable for internship.

Usage:
    python final_project.py --input data/ecommerce_sales_sample.csv --output results/

What it does:
- Loads a CSV transaction dataset (expects columns like InvoiceNo, Description, Quantity,
  InvoiceDate, UnitPrice, CustomerID, Country).
- Cleans the data, computes TotalPrice, adds date features.
- Computes KPIs (total revenue, orders, customers, AOV, top products, monthly revenue, sales by country).
- Saves outputs: results/kpis.json, results/summary.md, and plots in results/.
- Designed as a single file for easy submission.

Dependencies:
- pandas, numpy, matplotlib

Author: Aditya Singh
"""

from pathlib import Path
import argparse
import logging
import sys
import json
from typing import Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Helpers
# -------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_json(obj: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    logging.info("Saved JSON: %s", path)

def save_text(text: str, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logging.info("Saved text: %s", path)

# -------------------------
# Data functions
# -------------------------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    logging.info("Loading CSV from %s", path)
    # attempt parse InvoiceDate; if not present, will load normally
    try:
        df = pd.read_csv(path, parse_dates=["InvoiceDate"], infer_datetime_format=True)
    except Exception:
        df = pd.read_csv(path)
    logging.info("Loaded rows=%d cols=%d", df.shape[0], df.shape[1])
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logging.info("Starting basic cleaning")
    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Parse InvoiceDate if exists
    if "InvoiceDate" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Drop rows missing critical fields
    for col in ("InvoiceNo", "InvoiceDate"):
        if col in df.columns:
            df = df.dropna(subset=[col])

    # Drop rows without customer (optional - many KPI needs CustomerID)
    if "CustomerID" in df.columns:
        df = df.dropna(subset=["CustomerID"])

    # Numeric conversions
    for col in ("Quantity", "UnitPrice"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in numeric cols
    numeric_cols = [c for c in ("Quantity", "UnitPrice") if c in df.columns]
    if numeric_cols:
        df = df.dropna(subset=numeric_cols)

    # Remove zero quantity rows (may include returns as negative numbers)
    if "Quantity" in df.columns:
        df = df[df["Quantity"] != 0]

    # Remove zero UnitPrice records
    if "UnitPrice" in df.columns:
        df = df[df["UnitPrice"].abs() > 0]

    # Compute TotalPrice
    if set(("Quantity", "UnitPrice")).issubset(df.columns):
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    else:
        df["TotalPrice"] = 0.0

    logging.info("Cleaning complete. Remaining rows=%d", len(df))
    return df

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "InvoiceDate" in df.columns:
        df["InvoiceYear"] = df["InvoiceDate"].dt.year
        df["InvoiceMonth"] = df["InvoiceDate"].dt.month
        df["InvoiceMonthName"] = df["InvoiceDate"].dt.strftime("%Y-%m")
        df["InvoiceWeekday"] = df["InvoiceDate"].dt.day_name()
    return df

# -------------------------
# Analysis
# -------------------------
def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    kpis: Dict[str, Any] = {}
    df = df.copy()

    kpis["total_revenue"] = float(df["TotalPrice"].sum()) if "TotalPrice" in df.columns else 0.0
    kpis["total_orders"] = int(df["InvoiceNo"].nunique()) if "InvoiceNo" in df.columns else 0
    kpis["total_customers"] = int(df["CustomerID"].nunique()) if "CustomerID" in df.columns else 0
    kpis["average_order_value"] = float(kpis["total_revenue"] / max(1, kpis["total_orders"]))

    if "Description" in df.columns and "TotalPrice" in df.columns:
        prod = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False)
        kpis["top_products"] = prod.head(10).to_dict()
    else:
        kpis["top_products"] = {}

    if "InvoiceMonthName" in df.columns:
        monthly = df.groupby("InvoiceMonthName")["TotalPrice"].sum().sort_index()
        kpis["monthly_revenue"] = monthly.to_dict()
    else:
        kpis["monthly_revenue"] = {}

    if "Country" in df.columns:
        country = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
        kpis["sales_by_country"] = country.to_dict()
    else:
        kpis["sales_by_country"] = {}

    logging.info("KPIs computed")
    return kpis

def top_n_products_df(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(n).reset_index()

def monthly_revenue_df(df: pd.DataFrame) -> pd.DataFrame:
    if "InvoiceMonthName" not in df.columns and "InvoiceDate" in df.columns:
        df = df.assign(InvoiceMonthName=df["InvoiceDate"].dt.strftime("%Y-%m"))
    result = df.groupby("InvoiceMonthName")["TotalPrice"].sum().sort_index().reset_index()
    result.columns = ["InvoiceMonthName", "TotalPrice"]
    return result

# -------------------------
# Visualization
# -------------------------
def plot_monthly_revenue(monthly_df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    if monthly_df.empty:
        logging.warning("Monthly revenue dataframe is empty. Skipping monthly plot.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_df["InvoiceMonthName"], monthly_df["TotalPrice"], marker="o")
    plt.title("Monthly Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = out_dir / "monthly_revenue.png"
    plt.savefig(out)
    plt.close()
    logging.info("Saved plot: %s", out)

def plot_top_products(top_df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    if top_df.empty:
        logging.warning("Top products df empty. Skipping top products plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["Description"].iloc[::-1], top_df["TotalPrice"].iloc[::-1])
    plt.title("Top Products by Revenue")
    plt.xlabel("Revenue")
    plt.tight_layout()
    out = out_dir / "top_products.png"
    plt.savefig(out)
    plt.close()
    logging.info("Saved plot: %s", out)

def plot_sales_by_country(series: pd.Series, out_dir: Path, top_n: int = 10):
    ensure_dir(out_dir)
    if series.empty:
        logging.warning("Country series empty. Skipping country plot.")
        return
    top = series.head(top_n)
    plt.figure(figsize=(8, 6))
    plt.pie(top, labels=top.index, autopct="%1.1f%%", startangle=140)
    plt.title(f"Top {top_n} Countries by Revenue")
    plt.tight_layout()
    out = out_dir / "sales_by_country.png"
    plt.savefig(out)
    plt.close()
    logging.info("Saved plot: %s", out)

# -------------------------
# Main
# -------------------------
def main(input_path: Path, output_dir: Path):
    setup_logging()
    logging.info("Starting final_project.py")
    ensure_dir(output_dir)

    # Load
    df = load_csv(input_path)

    # Clean
    df = basic_cleaning(df)

    # Features
    df = add_date_features(df)

    # Analysis
    kpis = compute_kpis(df)
    save_json(kpis, output_dir / "kpis.json")

    # Summary markdown
    summary = [
        "# Analysis Summary",
        "",
        f"**Total revenue:** {kpis.get('total_revenue', 0):.2f}",
        f"**Total orders:** {kpis.get('total_orders', 0)}",
        f"**Total customers:** {kpis.get('total_customers', 0)}",
        f"**Average order value:** {kpis.get('average_order_value', 0):.2f}",
        "",
        "## Top products (top 10)",
    ]
    for prod, rev in kpis.get("top_products", {}).items():
        summary.append(f"- {prod}: {rev:.2f}")

    save_text("\n".join(summary), output_dir / "summary.md")

    # Visualizations
    monthly_df = monthly_revenue_df(df)
    plot_monthly_revenue(monthly_df, output_dir)

    top_df = top_n_products_df(df, n=10)
    plot_top_products(top_df, output_dir)

    if "Country" in df.columns:
        country_series = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
        plot_sales_by_country(country_series, output_dir)

    logging.info("All done. Results are in: %s", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-commerce Sales Data Analysis - single-file")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=False, default="results", help="Output directory for results")
    args = parser.parse_args()
    main(Path(args.input), Path(args.output))
