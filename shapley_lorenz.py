import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone


try:
    from shapley_lz import ShapleyLorenzShare
except ImportError:
    from shapley_lz.explainer.shapley_lz import ShapleyLorenzShare


def _xgb_dtype(dtype):
    if isinstance(dtype, pd.CategoricalDtype):
        return dtype
    if pd.api.types.is_integer_dtype(dtype):
        return np.int32
    if pd.api.types.is_float_dtype(dtype):
        return np.float32
    if pd.api.types.is_bool_dtype(dtype):
        return bool
    return dtype


def _slz_numeric_frame(X, feature_names, feature_dtypes):
    X_num = pd.DataFrame(index=X.index)

    for col in feature_names:
        dtype = feature_dtypes[col]
        if isinstance(dtype, pd.CategoricalDtype):
            X_num[col] = X[col].cat.codes.astype(np.int32)
        else:
            X_num[col] = X[col].astype(dtype)

    return X_num


def _sample_with_both_classes(X, y, n, random_state):
    y = pd.Series(y, index=X.index)
    n = min(n, len(X))

    sample = X.sample(n=n, random_state=random_state)
    if y.loc[sample.index].nunique() == y.nunique():
        return sample, y.loc[sample.index]

    class_idx = y.groupby(y).sample(n=1, random_state=random_state).index
    remaining_n = max(0, n - len(class_idx))

    if remaining_n > 0:
        remaining_pool = X.drop(index=class_idx, errors="ignore")
        remaining_idx = remaining_pool.sample(
            n=min(remaining_n, len(remaining_pool)),
            random_state=random_state,
        ).index
        sample_idx = class_idx.union(remaining_idx)
    else:
        sample_idx = class_idx

    return X.loc[sample_idx], y.loc[sample_idx]


def _decode_model_frame(X_input, feature_names, feature_dtypes):
    if isinstance(X_input, pd.DataFrame):
        X_df = X_input.copy().reindex(columns=feature_names)
    else:
        X_array = np.asarray(X_input)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        X_df = pd.DataFrame(X_array, columns=feature_names)

    for col, dtype in feature_dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype):
            codes = pd.to_numeric(X_df[col], errors="coerce").fillna(-1).to_numpy()
            codes = np.rint(codes).astype(np.int32)
            codes = np.clip(codes, -1, len(dtype.categories) - 1)
            X_df[col] = pd.Categorical.from_codes(
                codes,
                categories=dtype.categories,
                ordered=dtype.ordered,
            )
        else:
            X_df[col] = X_df[col].astype(dtype)

    return X_df


def _extract_values(slz_values, class_index=1):
    if isinstance(slz_values, (list, tuple)):
        values = np.asarray(slz_values[class_index])
    else:
        values = np.asarray(slz_values)

    if values.ndim == 2 and values.shape[1] >= 2:
        values = values[:, -1]
    else:
        values = values.ravel()

    return values.astype(float)


def estimate_shapley_lorenz_importance(
    base_model,
    X_train,
    y_train,
    X_eval_source,
    y_eval_source,
    features,
    background_n=30,
    eval_n=30,
    n_iter=250,
    class_index=1,
    random_state=42,
):
    """
    Fit a reduced model and estimate Shapley-Lorenz values for selected features.

    shapley_lz expects numeric NumPy arrays, while XGBoost can need categorical
    pandas dtypes. This helper encodes categories for shapley_lz and decodes
    them inside the prediction wrapper before calling the model.
    """
    feature_names = list(features)
    if not feature_names:
        raise ValueError("features must contain at least one column")

    X_train_slz = X_train[feature_names].copy()
    X_eval_slz_source = X_eval_source[feature_names].copy()

    slz_model = clone(base_model)
    slz_model.fit(X_train_slz, y_train)

    feature_dtypes = {
        col: _xgb_dtype(dtype)
        for col, dtype in X_train_slz.dtypes.items()
    }

    if not hasattr(np, "int"):
        np.int = int

    def predict_proba_wrapper(X_input):
        X_df = _decode_model_frame(X_input, feature_names, feature_dtypes)
        return slz_model.predict_proba(X_df)

    X_background, y_background = _sample_with_both_classes(
        X_train_slz,
        y_train,
        n=background_n,
        random_state=random_state,
    )
    X_background_slz = _slz_numeric_frame(
        X_background,
        feature_names,
        feature_dtypes,
    )

    X_eval, y_eval = _sample_with_both_classes(
        X_eval_slz_source,
        y_eval_source,
        n=eval_n,
        random_state=random_state,
    )
    X_eval_slz = _slz_numeric_frame(
        X_eval,
        feature_names,
        feature_dtypes,
    )

    slz = ShapleyLorenzShare(
        predict_proba_wrapper,
        X_background_slz.to_numpy(dtype=np.float32),
        np.asarray(y_background),
    )

    raw_values = slz.shapleyLorenz_val(
        X_eval_slz.to_numpy(dtype=np.float32),
        np.asarray(y_eval),
        n_iter=n_iter,
        class_prob=True,
        pred_out="predict_proba",
    )

    values = _extract_values(raw_values, class_index=class_index)
    importance = (
        pd.DataFrame({
            "feature": feature_names,
            "shapley_lorenz_value": values,
        })
        .sort_values("shapley_lorenz_value", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "importance": importance,
        "raw_values": raw_values,
        "model": slz_model,
        "features": feature_names,
        "background": X_background,
        "eval": X_eval,
    }


def plot_shapley_lorenz(slz_df, top_n=30):
    plot_df = (
        slz_df
        .head(top_n)
        .sort_values("shapley_lorenz_value", ascending=True)
    )

    plt.figure(figsize=(10, max(5, 0.35 * len(plot_df))))
    plt.barh(
        plot_df["feature"],
        plot_df["shapley_lorenz_value"],
    )
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Shapley-Lorenz value")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Shapley-Lorenz Feature Contributions")
    plt.tight_layout()
    plt.show()
