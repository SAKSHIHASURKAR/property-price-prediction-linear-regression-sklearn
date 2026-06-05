"""
visualization.py
----------------
Produces bar charts comparing model performance metrics
(Test R² and MSE) across all trained models.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """
    Draw side-by-side bar charts for Test R² and MSE.

    Args:
        results_df: Summary DataFrame produced by analysis.run_all_models().
                    Must contain columns: 'Model', 'Test_R2', 'MSE'.
        save_path:  If provided, save the figure to this file path
                    (e.g. 'outputs/model_comparison.png').
                    If None, the figure is displayed interactively.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Linear Regression Model Comparison", fontsize=14, fontweight='bold')

    # --- Test R² ---
    sns.barplot(
        ax=axes[0],
        x='Model',
        y='Test_R2',
        hue='Model',
        data=results_df,
        palette='viridis',
        legend=False,
    )
    axes[0].set_title('Test R² by Model')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Test R²')
    axes[0].set_ylim(0, 1.05)

    for bar in axes[0].patches:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=8,
        )

    # --- MSE ---
    sns.barplot(
        ax=axes[1],
        x='Model',
        y='MSE',
        hue='Model',
        data=results_df,
        palette='plasma',
        legend=False,
    )
    axes[1].set_title('MSE by Model')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Mean Squared Error')

    for bar in axes[1].patches:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height + (max(results_df['MSE']) * 0.01),
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[visualization] Figure saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_feature_importance(results_df: pd.DataFrame) -> None:
    """
    Simple bar chart showing how Test R² changes as features are removed.
    Useful for communicating feature importance at a glance.

    Args:
        results_df: Summary DataFrame sorted by Test_R2 (descending).
    """
    plt.figure(figsize=(8, 4))
    sns.barplot(
        x='Number of Features',
        y='Test_R2',
        hue='Model',
        data=results_df,
        palette='coolwarm',
        legend=True,
    )
    plt.title('Test R² vs. Number of Features Used')
    plt.xlabel('Number of Features')
    plt.ylabel('Test R²')
    plt.tight_layout()
    plt.show()
