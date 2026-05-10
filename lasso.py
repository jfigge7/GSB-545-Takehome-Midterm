import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_lasso_feature_screen(
    X_train,
    y_train,
    cv=5,
    random_state=42,
    max_iter=20000,
    n_jobs=None,
    fallback_top_n=5,
):
    """
    Fit a LassoCV model and summarize sparse coefficients by source feature.

    Categorical variables are one-hot encoded for the linear model, then mapped
    back to the original source columns for downstream feature screening.
    """
    num_cols = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = X_train.select_dtypes(
        include=["category", "object", "string"]
    ).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", _make_one_hot_encoder(), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Pipeline([
        ("preprocess", preprocess),
        (
            "model",
            LassoCV(
                cv=cv,
                random_state=random_state,
                max_iter=max_iter,
                n_jobs=n_jobs,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    feature_names = list(model.named_steps["preprocess"].get_feature_names_out())
    cat_encoder = model.named_steps["preprocess"].named_transformers_["cat"]

    source_features = num_cols.copy()
    for col, categories in zip(cat_cols, cat_encoder.categories_):
        source_features.extend([col] * len(categories))

    coef_df = (
        pd.DataFrame({
            "feature": feature_names,
            "source_feature": source_features,
            "coefficient": model.named_steps["model"].coef_.ravel(),
        })
        .assign(abs_coefficient=lambda x: x["coefficient"].abs())
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )

    feature_summary = (
        coef_df
        .groupby("source_feature", as_index=False)
        .agg(
            max_coefficient=("coefficient", "max"),
            min_coefficient=("coefficient", "min"),
            max_abs_coefficient=("abs_coefficient", "max"),
            nonzero_coefficients=("coefficient", lambda s: (s != 0).sum()),
            positive_coefficients=("coefficient", lambda s: (s > 0).sum()),
        )
        .sort_values("max_coefficient", ascending=False)
        .reset_index(drop=True)
    )

    nonzero_sources = set(
        coef_df.loc[coef_df["coefficient"] != 0, "source_feature"]
    )
    positive_sources = set(
        coef_df.loc[coef_df["coefficient"] > 0, "source_feature"]
    )

    nonzero_features = [
        col for col in X_train.columns
        if col in nonzero_sources
    ]
    positive_features = [
        col for col in X_train.columns
        if col in positive_sources
    ]

    if not positive_features:
        positive_features = feature_summary.head(fallback_top_n)["source_feature"].tolist()

    return {
        "model": model,
        "coef_df": coef_df,
        "feature_summary": feature_summary,
        "nonzero_features": nonzero_features,
        "positive_features": positive_features,
        "alpha": model.named_steps["model"].alpha_,
        "n_nonzero_encoded": int((coef_df["coefficient"] != 0).sum()),
        "n_encoded_features": len(coef_df),
    }


def plot_lasso_coefficients(coef_df, top_n=30):
    plot_df = (
        coef_df
        .head(top_n)
        .sort_values("abs_coefficient", ascending=True)
    )

    plt.figure(figsize=(10, max(5, 0.35 * len(plot_df))))
    plt.barh(
        plot_df["feature"],
        plot_df["abs_coefficient"],
    )
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Absolute Lasso coefficient value")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Lasso Regression Coefficient Contributions")
    plt.tight_layout()
    plt.show()
