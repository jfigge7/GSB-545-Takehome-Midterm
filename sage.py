import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _predict_proba_positive(model, X):
    """
    Returns predicted probability for class 1.
    Assumes binary classification.
    """
    proba = model.predict_proba(X)
    return proba[:, 1]


def _binary_log_loss_per_row(y_true, p_pred, eps=1e-15):
    """
    Per-row binary log loss.
    Lower is better.
    """
    y_true = np.asarray(y_true).astype(float)
    p_pred = np.clip(np.asarray(p_pred), eps, 1 - eps)

    return -(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))


def estimate_sage_importance(
    model,
    X_eval,
    y_eval,
    X_background,
    features=None,
    n_permutations=100,
    batch_size=512,
    random_state=42
):
    """
    Approximate SAGE values using Monte Carlo feature permutations.

    Interpretation:
        Positive value = feature reduces loss, so it helps the model.
        Near zero      = little unique global contribution.
        Negative value = feature tends to hurt performance under this loss.

    This uses marginal background replacement:
        - Start with all features replaced by values from random background rows.
        - Reveal features one at a time according to random permutations.
        - Measure the loss reduction when each feature is revealed.

    Parameters
    ----------
    model : fitted estimator with predict_proba
    X_eval : pd.DataFrame
        Data to explain, usually validation/test sample.
    y_eval : array-like
        Labels for X_eval.
    X_background : pd.DataFrame
        Background data used to impute hidden features.
    features : list[str], optional
        Features to explain. Defaults to all columns.
    n_permutations : int
        More permutations = lower Monte Carlo noise, higher runtime.
    batch_size : int
        Number of rows per batch.
    random_state : int

    Returns
    -------
    pd.DataFrame with SAGE values.
    """
    rng = np.random.default_rng(random_state)

    X_eval = X_eval.copy()
    X_background = X_background.copy()
    y_eval = pd.Series(y_eval).reset_index(drop=True)

    if features is None:
        features = list(X_eval.columns)

    n_features = len(features)
    n_rows = len(X_eval)

    sage_values = np.zeros(n_features, dtype=float)
    sage_sq_values = np.zeros(n_features, dtype=float)
    contribution_counts = np.zeros(n_features, dtype=int)

    feature_to_idx = {feature: i for i, feature in enumerate(features)}

    # Work in batches so this doesn't explode memory
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)

        X_batch = X_eval.iloc[start:end].copy()
        y_batch = y_eval.iloc[start:end].to_numpy()
        m = len(X_batch)

        for _ in range(n_permutations):
            # Sample full background rows. This preserves joint structure among
            # hidden features better than sampling each column independently.
            bg_idx = rng.integers(0, len(X_background), size=m)
            X_current = X_background.iloc[bg_idx].reset_index(drop=True).copy()

            # Keep category dtypes aligned with original X_eval
            for col in X_eval.columns:
                if isinstance(X_eval[col].dtype, pd.CategoricalDtype):
                    X_current[col] = X_current[col].astype(X_eval[col].dtype)

            permutation = rng.permutation(features)

            p_current = _predict_proba_positive(model, X_current)
            loss_current = _binary_log_loss_per_row(y_batch, p_current)

            for feature in permutation:
                X_next = X_current.copy()
                X_next[feature] = X_batch[feature].reset_index(drop=True)

                if isinstance(X_eval[feature].dtype, pd.CategoricalDtype):
                    X_next[feature] = X_next[feature].astype(X_eval[feature].dtype)

                p_next = _predict_proba_positive(model, X_next)
                loss_next = _binary_log_loss_per_row(y_batch, p_next)

                # Loss reduction from revealing this feature
                contribution = loss_current - loss_next

                j = feature_to_idx[feature]
                sage_values[j] += contribution.mean()
                sage_sq_values[j] += contribution.mean() ** 2
                contribution_counts[j] += 1

                X_current = X_next
                loss_current = loss_next

    sage_mean = sage_values / contribution_counts

    # Rough Monte Carlo standard error across permutation-level contributions
    sage_var = (sage_sq_values / contribution_counts) - sage_mean ** 2
    sage_se = np.sqrt(np.maximum(sage_var, 0) / contribution_counts)

    result = pd.DataFrame({
        "feature": features,
        "sage_value": sage_mean,
        "sage_se": sage_se,
        "sage_abs": np.abs(sage_mean)
    })

    result = result.sort_values("sage_value", ascending=False).reset_index(drop=True)

    return result


def plot_sage_importance(sage_df, top_n=20):
    plot_df = (
        sage_df
        .head(top_n)
        .sort_values("sage_abs", ascending=True)
    )

    plt.figure(figsize=(10, max(5, 0.35 * len(plot_df))))
    plt.barh(
        plot_df["feature"],
        plot_df["sage_value"],
    )
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("SAGE value")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Shapley Global Importance (log-loss) Contributions")
    plt.tight_layout()
    plt.show()
