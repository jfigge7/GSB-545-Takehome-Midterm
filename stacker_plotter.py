import numpy as np
import matplotlib.pyplot as plt

def plot_stack_coefs(ensemble):
    stack_model = ensemble

    # The final estimator is GridSearchCV(LogisticRegression), so unwrap the best model.
    meta_model = stack_model.final_estimator_
    if hasattr(meta_model, "best_estimator_"):
        meta_model = meta_model.best_estimator_

    coef = np.ravel(meta_model.coef_)

    try:
        base_model_names = stack_model.get_feature_names_out()
    except Exception:
        base_model_names = [name for name, _ in stack_model.estimators]

    if len(base_model_names) != len(coef):
        base_model_names = [name for name, _ in stack_model.estimators]

    if len(base_model_names) != len(coef):
        base_model_names = [f"meta_feature_{i}" for i in range(len(coef))]

    stack_coef_df = (
        pd.DataFrame({
            "base_model": base_model_names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        })
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )

    top_n = 25
    plot_df = stack_coef_df.head(top_n).sort_values("abs_coefficient", ascending=True)
    colors = np.where(plot_df["coefficient"] >= 0, "tab:blue", "tab:red")

    plt.figure(figsize=(10, max(6, 0.35 * len(plot_df))))
    plt.barh(plot_df["base_model"], plot_df["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Logistic regression meta-learner coefficient")
    plt.ylabel("Base model")
    plt.title(f"Top {top_n} StackingClassifier Coefficients")
    plt.tight_layout()
    plt.show()