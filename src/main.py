"""
main.py
-------
Entry point for the Property Price Prediction project.

Pipeline:
  1. Load dataset          → data_loader
  2. Clean & validate data → data_cleaning
  3. Train & evaluate      → analysis
  4. Visualise results     → visualization

Usage:
    python main.py
    python main.py --data path/to/property.csv --save-fig outputs/comparison.png
"""

import argparse
import os

from data_loader import load_data, preview_data
from data_cleaning import clean_data
from analysis import run_all_models
from visualization import plot_model_comparison, plot_feature_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Property Price Prediction — Multiple Linear Regression"
    )
    parser.add_argument(
        '--data',
        type=str,
        default='property (1).csv',
        help='Path to the input CSV file (default: "property (1).csv")',
    )
    parser.add_argument(
        '--save-fig',
        type=str,
        default=None,
        help='Optional path to save the comparison chart (e.g. outputs/chart.png)',
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip all visualisations (useful in headless environments)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)
    df_raw = load_data(args.data)
    preview_data(df_raw)

    # ── 2. Clean ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Cleaning data")
    print("=" * 60)
    df_clean = clean_data(df_raw)

    # ── 3. Analyse ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Training & evaluating models")
    print("=" * 60)
    results_df, raw_results = run_all_models(df_clean)

    # ── 4. Visualise ─────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n" + "=" * 60)
        print("STEP 4: Visualising results")
        print("=" * 60)

        # Ensure output directory exists if saving
        if args.save_fig:
            os.makedirs(os.path.dirname(args.save_fig) or '.', exist_ok=True)

        plot_model_comparison(results_df, save_path=args.save_fig)
        plot_feature_importance(results_df)

    print("\n✓ Pipeline complete.")


if __name__ == '__main__':
    main()
