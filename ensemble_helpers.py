from dataclasses import dataclass

import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


FOCUSED_FEATURES = [
    "contact_time_minutes",
    "last_contact_month",
    "reference_interest_rate",
    "prior_outcome_status",
]

EDUCATION_ORDER = [
    "illiterate",
    "basic.4y",
    "basic.6y",
    "basic.9y",
    "high.school",
    "professional.course",
    "university.degree",
    "unknown",
]

f1_class_0 = make_scorer(f1_score, pos_label=0)
f1_class_1 = make_scorer(f1_score, pos_label=1)
f1_macro_scorer = make_scorer(f1_score, average="macro", zero_division=0)


@dataclass
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict
    scoring: object
    report_metric: str
    family: str


class FeatureSelector(BaseEstimator):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, list(self.features)].copy()


class RegressorAsClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, regressor=None, threshold=0.5, eps=1e-6):
        self.regressor = regressor
        self.threshold = threshold
        self.eps = eps

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y)
        return self

    def predict_proba(self, X):
        pred = np.asarray(self.regressor_.predict(X), dtype=float)
        pred = np.clip(pred, self.eps, 1 - self.eps)
        return np.column_stack([1 - pred, pred])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator=None, threshold=0.5):
        self.base_estimator = base_estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y)
        self.classes_ = getattr(self.estimator_, "classes_", np.array([0, 1]))
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        positive_index = np.where(self.classes_ == 1)[0]
        if len(positive_index) == 0:
            positive_index = [proba.shape[1] - 1]

        return (proba[:, positive_index[0]] >= self.threshold).astype(int)


class CatBoostFrameCleaner(BaseEstimator):
    def __init__(self, categorical_features=None):
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in list(self.categorical_features or []):
            if col in X.columns:
                X[col] = X[col].astype("string").fillna("__missing__").astype(str)
        return X


class NativeCatBoostClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        categorical_features=None,
        loss_function="Logloss",
        eval_metric=None,
        task_type="GPU",
        devices="0",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        iterations=None,
        depth=None,
        learning_rate=None,
        auto_class_weights=None,
    ):
        self.categorical_features = categorical_features
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.task_type = task_type
        self.devices = devices
        self.random_seed = random_seed
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.auto_class_weights = auto_class_weights

    def fit(self, X, y):
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed.")

        params = {
            "loss_function": self.loss_function,
            "task_type": self.task_type,
            "devices": self.devices,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "allow_writing_files": self.allow_writing_files,
        }
        optional_params = {
            "eval_metric": self.eval_metric,
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "auto_class_weights": self.auto_class_weights,
        }
        params.update({
            key: value
            for key, value in optional_params.items()
            if value is not None
        })

        self.estimator_ = CatBoostClassifier(**params)
        self.estimator_.fit(
            X,
            y,
            cat_features=list(self.categorical_features or []),
        )
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X)).astype(int).ravel()


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _categorical_columns(X, features):
    return [
        col for col in features
        if (
            isinstance(X[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(X[col].dtype)
            or pd.api.types.is_string_dtype(X[col].dtype)
        )
    ]


def _numeric_columns(X, features):
    return [col for col in features if col not in _categorical_columns(X, features)]


def _education_categories(X):
    if "education_background" not in X.columns:
        return [EDUCATION_ORDER]

    seen = set(X["education_background"].astype(str).unique())
    extras = sorted(seen.difference(EDUCATION_ORDER))
    return [EDUCATION_ORDER + extras]


def build_preprocessor(X, features, kind, education_encoding="onehot", polynomial=False):
    numeric_cols = _numeric_columns(X, features)
    categorical_cols = _categorical_columns(X, features)
    transformers = []

    if kind == "tree":
        if numeric_cols:
            transformers.append(("num", "passthrough", numeric_cols))
        if categorical_cols:
            transformers.append((
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_cols,
            ))
    else:
        if numeric_cols:
            if polynomial:
                numeric_transformer = Pipeline([
                    ("scale", StandardScaler()),
                    (
                        "poly",
                        PolynomialFeatures(
                            degree=2,
                            interaction_only=True,
                            include_bias=False,
                        ),
                    ),
                ])
            else:
                numeric_transformer = StandardScaler()

            transformers.append(("num", numeric_transformer, numeric_cols))

        edu_cols = [col for col in categorical_cols if col == "education_background"]
        other_cat_cols = [col for col in categorical_cols if col != "education_background"]

        if education_encoding == "ordinal" and edu_cols:
            transformers.append((
                "education",
                OrdinalEncoder(
                    categories=_education_categories(X),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                edu_cols,
            ))
            if other_cat_cols:
                transformers.append(("cat", _make_one_hot_encoder(), other_cat_cols))
        elif categorical_cols:
            transformers.append(("cat", _make_one_hot_encoder(), categorical_cols))

    steps = [(
        "preprocess",
        ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        ),
    )]

    return Pipeline(steps)


def _sampler_step(sampling, random_state):
    if sampling == "undersample":
        return ("sample", RandomUnderSampler(random_state=random_state))
    if sampling == "oversample":
        return ("sample", RandomOverSampler(random_state=random_state))
    return None


def make_pipeline_for_spec(
    X,
    features,
    estimator,
    kind,
    sampling,
    education_encoding="onehot",
    polynomial=False,
    random_state=42,
):
    preprocessing = build_preprocessor(
        X,
        features=features,
        kind=kind,
        education_encoding=education_encoding,
        polynomial=polynomial,
    )

    steps = [("select", FeatureSelector(features))]
    steps.extend(preprocessing.steps)

    sampler = _sampler_step(sampling, random_state)
    if sampler is not None:
        steps.append(sampler)

    steps.append(("estimator", estimator))
    return ImbPipeline(steps)


def make_catboost_pipeline_for_spec(
    X,
    features,
    estimator,
    sampling,
    random_state=42,
):
    categorical_features = _categorical_columns(X, features)

    steps = [
        ("select", FeatureSelector(features)),
        ("catboost_prepare", CatBoostFrameCleaner(categorical_features)),
    ]

    sampler = _sampler_step(sampling, random_state)
    if sampler is not None:
        steps.append(sampler)

    steps.append(("estimator", estimator))
    return ImbPipeline(steps)


def _safe_name(*parts):
    return "_".join(str(part).replace("-", "_") for part in parts if part is not None)


def _feature_sets(X, feature_sets):
    if feature_sets is not None:
        return {key: list(value) for key, value in feature_sets.items()}

    focused = [col for col in FOCUSED_FEATURES if col in X.columns]
    return {
        "all": list(X.columns),
        "focused": focused,
    }


def _scale_pos_weight_grid(y):
    if y is None:
        return [1.0, 3.0, 6.0]

    counts = pd.Series(y).value_counts()
    negative = counts.get(0, 0)
    positive = counts.get(1, 0)
    if negative == 0 or positive == 0:
        return [1.0]

    ratio = negative / positive
    values = [1.0, ratio / 2, ratio]
    return sorted({round(float(max(1.0, value)), 3) for value in values})


def build_model_specs(X, feature_sets=None, random_state=42, y=None):
    feature_sets = _feature_sets(X, feature_sets)
    sampling_options = ["none", "undersample", "oversample"]
    specs = []

    xgb_class_0_grid = {
        "estimator__n_estimators": [150, 300],
        "estimator__max_depth": [2, 3, 4],
        "estimator__learning_rate": [0.03, 0.06],
    }
    xgb_class_1_grid = {
        "estimator__base_estimator__n_estimators": [150, 300],
        "estimator__base_estimator__max_depth": [2, 3],
        "estimator__base_estimator__learning_rate": [0.03, 0.06],
        "estimator__base_estimator__scale_pos_weight": _scale_pos_weight_grid(y),
        "estimator__threshold": [0.35, 0.45, 0.5],
    }
    xgb_f1_macro_grid = {
        "estimator__base_estimator__n_estimators": [150, 300],
        "estimator__base_estimator__max_depth": [2, 3],
        "estimator__base_estimator__learning_rate": [0.03, 0.06],
        "estimator__base_estimator__scale_pos_weight": _scale_pos_weight_grid(y),
        "estimator__threshold": [0.4, 0.5, 0.6],
    }
    xgb_param_grids = {
        "f1_class_0": xgb_class_0_grid,
        "f1_class_1": xgb_class_1_grid,
        "f1_macro": xgb_f1_macro_grid,
    }
    catboost_balanced_grid = {
        "estimator__iterations": [200, 400],
        "estimator__depth": [4, 6],
        "estimator__learning_rate": [0.03, 0.08],
        "estimator__auto_class_weights": [None, "Balanced"],
    }
    catboost_f1_macro_grid = {
        "estimator__base_estimator__iterations": [200, 400],
        "estimator__base_estimator__depth": [4, 6],
        "estimator__base_estimator__learning_rate": [0.05],
        "estimator__base_estimator__auto_class_weights": [None, "Balanced"],
        "estimator__threshold": [0.4, 0.5, 0.6],
    }

    for feature_set_name, features in feature_sets.items():
        for sampling in sampling_options:
            for score_name, scorer, report_metric in [
                ("f1_class_0", f1_class_0, "f1_class_0"),
                ("f1_class_1", f1_class_1, "f1_class_1"),
                ("f1_macro", f1_macro_scorer, "f1_macro"),
            ]:
                estimator = XGBClassifier(
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=1,
                    tree_method="hist",
                    device="cuda",
                )
                if score_name in ("f1_class_1", "f1_macro"):
                    estimator = ThresholdClassifier(estimator)

                specs.append(ModelSpec(
                    name=_safe_name("xgb", score_name, feature_set_name, sampling),
                    estimator=make_pipeline_for_spec(
                        X, features, estimator, "tree", sampling,
                        random_state=random_state,
                    ),
                    param_grid=xgb_param_grids[score_name],
                    scoring=scorer,
                    report_metric=report_metric,
                    family="xgboost",
                ))

            estimator = ExtraTreesClassifier(
                random_state=random_state,
                n_jobs=1,
                class_weight="balanced",
            )
            specs.append(ModelSpec(
                name=_safe_name("extra_trees", feature_set_name, sampling),
                estimator=make_pipeline_for_spec(
                    X, features, estimator, "tree", sampling,
                    random_state=random_state,
                ),
                param_grid={
                    "estimator__n_estimators": [100],
                    "estimator__max_depth": [None, 6],
                },
                scoring="balanced_accuracy",
                report_metric="balanced_accuracy",
                family="extra_trees",
            ))

            if CatBoostClassifier is not None:
                categorical_features = _categorical_columns(X, features)

                estimator = NativeCatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    categorical_features=categorical_features,
                    task_type="GPU",
                    devices="0",
                    random_seed=random_state,
                    verbose=False,
                    allow_writing_files=False,
                )
                specs.append(ModelSpec(
                    name=_safe_name("catboost", "balanced_accuracy", feature_set_name, sampling),
                    estimator=make_catboost_pipeline_for_spec(
                        X, features, estimator, sampling,
                        random_state=random_state,
                    ),
                    param_grid=catboost_balanced_grid,
                    scoring="balanced_accuracy",
                    report_metric="balanced_accuracy",
                    family="catboost",
                ))

                estimator = ThresholdClassifier(NativeCatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    categorical_features=categorical_features,
                    task_type="GPU",
                    devices="0",
                    random_seed=random_state,
                    verbose=False,
                    allow_writing_files=False,
                ))
                specs.append(ModelSpec(
                    name=_safe_name("catboost", "f1_macro", feature_set_name, sampling),
                    estimator=make_catboost_pipeline_for_spec(
                        X, features, estimator, sampling,
                        random_state=random_state,
                    ),
                    param_grid=catboost_f1_macro_grid,
                    scoring=f1_macro_scorer,
                    report_metric="f1_macro",
                    family="catboost",
                ))

    non_tree_models = [
        (
            "naive_bayes",
            GaussianNB(),
            {"estimator__var_smoothing": [1e-9, 1e-8]},
            "recall_macro",
        ),
        (
            "linear_regression",
            RegressorAsClassifier(LinearRegression()),
            {"estimator__regressor__fit_intercept": [True, False]},
            "precision_macro",
        ),
        (
            "logistic_regression",
            LogisticRegression(max_iter=2000, random_state=random_state),
            {"estimator__C": [0.5, 1.0]},
            "f1_macro",
        ),
        (
            "lasso_regression",
            RegressorAsClassifier(
                Lasso(
                    max_iter=3000,
                    random_state=random_state,
                    selection="random",
                    tol=1e-3,
                )
            ),
            {"estimator__regressor__alpha": [0.0005, 0.001, 0.005]},
            "balanced_accuracy",
        ),
    ]

    for feature_set_name, features in feature_sets.items():
        for sampling in sampling_options:
            for polynomial in [False, True]:
                for education_encoding in ["onehot", "ordinal"]:
                    for model_name, estimator, grid, report_metric in non_tree_models:
                        specs.append(ModelSpec(
                            name=_safe_name(
                                model_name,
                                feature_set_name,
                                sampling,
                                "poly" if polynomial else "linear",
                                f"edu_{education_encoding}",
                            ),
                            estimator=make_pipeline_for_spec(
                                X,
                                features,
                                estimator,
                                "non_tree",
                                sampling,
                                education_encoding=education_encoding,
                                polynomial=polynomial,
                                random_state=random_state,
                            ),
                            param_grid=grid,
                            scoring="balanced_accuracy",
                            report_metric=report_metric,
                            family=model_name,
                        ))

    return specs


def compute_metrics(y_true, y_pred):
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_class_0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_class_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_class_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_class_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_class_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_class_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def fit_model_zoo(
    specs,
    X_train,
    y_train,
    X_test,
    y_test,
    cv=None,
    n_jobs=1,
    verbose=True,
    max_models=None,
):
    if cv is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    selected_specs = specs if max_models is None else specs[:max_models]
    results = []

    for i, spec in enumerate(selected_specs, start=1):
        search = GridSearchCV(
            estimator=spec.estimator,
            param_grid=spec.param_grid,
            scoring=spec.scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        train_pred = best_estimator.predict(X_train)
        test_pred = best_estimator.predict(X_test)
        train_metrics = compute_metrics(y_train, train_pred)
        test_metrics = compute_metrics(y_test, test_pred)

        row = {
            "name": spec.name,
            "family": spec.family,
            "best_cv_score": search.best_score_,
            "best_params": search.best_params_,
            "report_metric": spec.report_metric,
            "train_report_metric": train_metrics[spec.report_metric],
            "test_report_metric": test_metrics[spec.report_metric],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "test_f1_macro": test_metrics["f1_macro"],
            "train_precision_class_0": train_metrics["precision_class_0"],
            "test_precision_class_0": test_metrics["precision_class_0"],
            "train_precision_class_1": train_metrics["precision_class_1"],
            "test_precision_class_1": test_metrics["precision_class_1"],
            "train_recall_class_0": train_metrics["recall_class_0"],
            "test_recall_class_0": test_metrics["recall_class_0"],
            "train_recall_class_1": train_metrics["recall_class_1"],
            "test_recall_class_1": test_metrics["recall_class_1"],
            "train_f1_class_0": train_metrics["f1_class_0"],
            "test_f1_class_0": test_metrics["f1_class_0"],
            "train_f1_class_1": train_metrics["f1_class_1"],
            "test_f1_class_1": test_metrics["f1_class_1"],
            "estimator": best_estimator,
            "search": search,
        }
        results.append(row)

        if verbose:
            print(
                f"[{i:03d}/{len(selected_specs):03d}] {spec.name} | "
                f"cv={search.best_score_:.4f} | "
                f"train_{spec.report_metric}={row['train_report_metric']:.4f} | "
                f"test_{spec.report_metric}={row['test_report_metric']:.4f}"
            )

    summary = (
        pd.DataFrame([{k: v for k, v in row.items() if k not in ("estimator", "search")} for row in results])
        .sort_values("test_balanced_accuracy", ascending=False)
        .reset_index(drop=True)
    )

    stacking_estimators = make_stacking_estimators(results)

    return results, summary, stacking_estimators


def make_stacking_estimators(results, top_n=None):
    rows = results
    if top_n is not None:
        rows = sorted(
            results,
            key=lambda row: row["test_balanced_accuracy"],
            reverse=True,
        )[:top_n]

    return [(row["name"], row["estimator"]) for row in rows]


def make_stacking_classifier(results, final_estimator=None, top_n=None):
    estimators = make_stacking_estimators(results, top_n=top_n)

    if final_estimator is None:
        final_estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )

    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method="predict_proba",
        cv="prefit",
        n_jobs=1,
    )


def print_metric_report(name, estimator, X_train, y_train, X_test, y_test):
    train_pred = estimator.predict(X_train)
    test_pred = estimator.predict(X_test)
    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    print(f"{name} train balanced accuracy: {train_metrics['balanced_accuracy']:.4f}")
    print(f"{name} test balanced accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    print(f"{name} train f1 macro:          {train_metrics['f1_macro']:.4f}")
    print(f"{name} test f1 macro:           {test_metrics['f1_macro']:.4f}")

    return {
        "train": train_metrics,
        "test": test_metrics,
    }
