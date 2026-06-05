"""
data_cleaning.py
----------------
Handles data validation, missing value checks, and cleaning
for the property price dataset.
"""

import pandas as pd


REQUIRED_COLUMNS = [
    'Square_Footage',
    'Num_Bedrooms',
    'Num_Bathrooms',
    'Year_Built',
    'Lot_Size',
    'Price',
]


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Report the count of missing values per column.

    Args:
        df: Raw DataFrame.

    Returns:
        Series with missing value counts per column.
    """
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("[data_cleaning] No missing values found.")
    else:
        print("[data_cleaning] Missing values detected:")
        print(missing[missing > 0])
    return missing


def validate_columns(df: pd.DataFrame) -> None:
    """
    Ensure all required columns are present in the DataFrame.
    Raises a ValueError if any are missing.

    Args:
        df: DataFrame to validate.
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[data_cleaning] Missing required columns: {missing_cols}"
        )
    print("[data_cleaning] All required columns are present.")


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"[data_cleaning] Removed {removed} duplicate row(s).")
    else:
        print("[data_cleaning] No duplicates found.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline:
      1. Validate required columns
      2. Check and drop rows with missing values
      3. Remove duplicates

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    validate_columns(df)
    check_missing_values(df)

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    after = len(df)
    if before != after:
        print(f"[data_cleaning] Dropped {before - after} row(s) with missing values.")

    df = drop_duplicates(df)
    print(f"[data_cleaning] Clean dataset shape: {df.shape}")
    return df
