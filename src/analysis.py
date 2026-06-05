"""
analysis.py
-----------
Defines feature sets, trains multiple Linear Regression models,
and evaluates their performance using MSE and R² metrics.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Five feature combinations — each drops one predictor to assess its importance
FEATURE_SETS = [
    ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size'],  # Model 1: All features
    ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built'],              # Model 2: No Lot_Size
    ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Lot_Size'],                # Model 3: No Year_Built
    ['Square_Footage', 'Num_Bedrooms', 'Year_Built', 'Lot_Size'],                   # Model 4: No Num_Bathrooms
    ['Square_Footage', 'Num_Bathrooms', 'Year_Built', 'Lot_Size'],                  # Model 5: No Num_Bedrooms
]

TARGET = 'Price'


def train_single_model(
    df: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train a Linear Regression model for one feature combination and
    return its performance metrics and coefficients.

    Args:
        df:           Cleaned DataFrame.
        features:     List of feature column names.
        target:       Target column name.
        test_size:    Proportion of data used for testing.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary containing model metadata, metrics, and coefficients.
    """
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2_train = model.score(X_train, y_train)
    r2_test = r2_score(y_test, y_pred)

    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    })

    return {
        'features': features,
        'model': model,
        'coef_df': coef_df,
        'mse': mse,
        'r2_train': r2_train,
        'r2_test': r2_test,
    }


def run_all_models(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Train and evaluate all five feature-set models.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Tuple of:
          - results_df: Sorted summary DataFrame (best Test R² first).
          - raw_results: List of per-model result dicts (for further inspection).
    """
    raw_results = []

    for i, features in enumerate(FEATURE_SETS, start=1):
        print(f"\nTraining Model {i} with features: {features}")
        result = train_single_model(df, features)
        result['model_name'] = f'Model {i}'

        print(f"  Coefficients:\n{result['coef_df'].to_string(index=False)}")
        print(f"  Train R²: {result['r2_train']:.4f}")
        print(f"  Test  R²: {result['r2_test']:.4f}")
        print(f"  MSE:      {result['mse']:.4f}")

        raw_results.append(result)

    # Build a flat summary DataFrame
    summary_rows = [
        {
            'Model': r['model_name'],
            'Features Used': ', '.join(r['features']),
            'Number of Features': len(r['features']),
            'MSE': r['mse'],
            'Train_R2': r['r2_train'],
            'Test_R2': r['r2_test'],
        }
        for r in raw_results
    ]

    results_df = (
        pd.DataFrame(summary_rows)
        .sort_values(by='Test_R2', ascending=False)
        .reset_index(drop=True)
    )

    print("\n--- Final Model Comparison (sorted by Test R²) ---")
    print(results_df.to_string(index=False))

    return results_df, raw_results
