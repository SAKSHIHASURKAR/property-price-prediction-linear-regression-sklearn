"""
data_loader.py
--------------
Handles loading and initial inspection of the property dataset.
"""

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the property CSV dataset from the given file path.

    Args:
        filepath: Path to the CSV file (e.g., 'property (1).csv')

    Returns:
        DataFrame containing the raw dataset.
    """
    df = pd.read_csv(filepath)
    print(f"[data_loader] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[data_loader] Columns: {list(df.columns)}")
    return df


def preview_data(df: pd.DataFrame, n: int = 5) -> None:
    """Print the first n rows and basic info of the DataFrame."""
    print("\n--- Data Preview ---")
    print(df.head(n))
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Basic Statistics ---")
    print(df.describe())
