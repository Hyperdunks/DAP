from __future__ import annotations

import io
from pathlib import Path


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install required packages with: "
        "pip install pandas numpy matplotlib seaborn scikit-learn"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "netflix_customer_churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = OUTPUT_DIR / "analysis_report.txt"


NUMERIC_COLUMNS = [
    "age",
    "watch_hours",
    "last_login_days",
    "monthly_fee",
    "churned",
    "number_of_profiles",
    "avg_watch_time_per_day",
]

REGRESSION_FEATURES = [
    "age",
    "last_login_days",
    "monthly_fee",
    "number_of_profiles",
    "avg_watch_time_per_day",
]

CLASSIFICATION_NUMERIC_FEATURES = [
    "age",
    "watch_hours",
    "last_login_days",
    "monthly_fee",
    "number_of_profiles",
    "avg_watch_time_per_day",
]

CLASSIFICATION_CATEGORICAL_FEATURES = [
    "gender",
    "subscription_type",
    "region",
    "device",
    "payment_method",
    "favorite_genre",
]


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)


def log(lines: list[str], message: str = "") -> None:
    print(message)
    lines.append(message)


def log_heading(lines: list[str], title: str) -> None:
    border = "=" * len(title)
    log(lines, "")
    log(lines, title)
    log(lines, border)


def dataframe_info_string(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(TABLES_DIR / filename, index=True)


def save_current_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def classify_columns() -> dict[str, list[str]]:
    return {
        "quantitative_discrete": ["age", "last_login_days", "number_of_profiles", "churned"],
        "quantitative_continuous": ["watch_hours", "monthly_fee", "avg_watch_time_per_day"],
        "qualitative_nominal": [
            "customer_id",
            "gender",
            "region",
            "device",
            "payment_method",
            "favorite_genre",
        ],
        "qualitative_ordinal": ["subscription_type"],
    }


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null_count": df.notnull().sum(),
            "missing_count": df.isnull().sum(),
            "unique_values": df.nunique(),
        }
    )
    return summary


def plot_histogram_and_boxplot(df: pd.DataFrame, column: str, bins: int = 25) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df[column], kde=True, bins=bins, ax=axes[0], color="#2563eb")
    axes[0].set_title(f"Histogram of {column}")
    sns.boxplot(x=df[column], ax=axes[1], color="#f59e0b")
    axes[1].set_title(f"Boxplot of {column}")
    save_current_figure(f"univariate_{column}.png")


def plot_categorical_count(df: pd.DataFrame, column: str) -> None:
    plt.figure(figsize=(10, 5))
    order = df[column].value_counts().index
    ax = sns.countplot(data=df, x=column, order=order, hue=column, palette="Blues_r", legend=False)
    plt.title(f"Count Plot of {column}")
    plt.xticks(rotation=25, ha="right")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    save_current_figure(f"countplot_{column}.png")


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue="churned", alpha=0.65, palette="Set1")
    plt.title(f"{y_col} vs {x_col}")
    save_current_figure(filename)


def plot_heatmap(df: pd.DataFrame, columns: list[str], filename: str) -> pd.DataFrame:
    corr = df[columns].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    save_current_figure(filename)
    return corr


def plot_pairplot(df: pd.DataFrame, columns: list[str], filename: str) -> None:
    sampled = df[columns + ["churned"]].sample(n=min(600, len(df)), random_state=42)
    pair_grid = sns.pairplot(sampled, vars=columns, hue="churned", corner=True, diag_kind="hist")
    pair_grid.figure.suptitle("Pair Plot of Key Numeric Features", y=1.02)
    pair_grid.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(pair_grid.figure)


def plot_grouped_boxplot(df: pd.DataFrame, x_col: str, y_col: str, filename: str) -> None:
    plt.figure(figsize=(11, 5))
    ax = sns.boxplot(data=df, x=x_col, y=y_col, hue=x_col, palette="Set3", dodge=False)
    plt.title(f"{y_col} across {x_col}")
    plt.xticks(rotation=25, ha="right")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    save_current_figure(filename)


def distribution_statistics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    stats_df = pd.DataFrame(
        {
            "mean": df[columns].mean(),
            "median": df[columns].median(),
            "std_dev": df[columns].std(),
            "skewness": df[columns].skew(),
            "kurtosis": df[columns].kurt(),
        }
    )
    return stats_df


def outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": int(mask.sum()),
                "outlier_percentage": round(mask.mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "MSE": mean_squared_error(y_true, predictions),
        "MAE": mean_absolute_error(y_true, predictions),
        "R2": r2_score(y_true, predictions),
    }


def classification_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, predictions),
    }


def save_confusion_heatmap(matrix: np.ndarray, filename: str, title: str) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    save_current_figure(filename)


def main() -> None:
    ensure_output_dirs()
    report_lines: list[str] = []

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    sns.set_theme(style="whitegrid")
    df = pd.read_csv(DATASET_PATH)

    log_heading(report_lines, "Streaming Data Analysis Using Netflix Customer Churn Dataset")
    log(report_lines, f"Dataset path: {DATASET_PATH}")
    log(report_lines, "Project note: the original rubric is retail-oriented, so streaming equivalents are used.")
    log(report_lines, "Sales equivalent: watch_hours | classification outcome: churned")

    log_heading(report_lines, "Task 1 - Data Understanding")
    log(report_lines, "First 5 rows:")
    log(report_lines, df.head().to_string(index=False))
    log(report_lines, "")
    log(report_lines, "Last 5 rows:")
    log(report_lines, df.tail().to_string(index=False))
    log(report_lines, "")
    log(report_lines, f"Dataset shape: {df.shape}")
    log(report_lines, f"Column names: {list(df.columns)}")

    info_text = dataframe_info_string(df)
    summary_table = build_summary_table(df)
    save_dataframe(summary_table, "column_summary.csv")
    log(report_lines, "")
    log(report_lines, "DataFrame info():")
    log(report_lines, info_text.strip())
    log(report_lines, "")
    log(report_lines, "describe() output:")
    describe_df = df[NUMERIC_COLUMNS].describe().round(3)
    save_dataframe(describe_df, "describe_numeric.csv")
    log(report_lines, describe_df.to_string())

    column_groups = classify_columns()
    for group_name, columns in column_groups.items():
        log(report_lines, f"{group_name}: {columns}")

    log_heading(report_lines, "Task 2 - Exploratory Data Analysis")
    for column in ["watch_hours", "monthly_fee", "avg_watch_time_per_day", "last_login_days"]:
        plot_histogram_and_boxplot(df, column)
    for column in ["subscription_type", "device", "favorite_genre", "churned"]:
        plot_categorical_count(df, column)

    plot_scatter(df, "avg_watch_time_per_day", "watch_hours", "scatter_watch_hours_vs_avg_watch_time_per_day.png")
    plot_scatter(df, "last_login_days", "watch_hours", "scatter_watch_hours_vs_last_login_days.png")
    plot_scatter(df, "monthly_fee", "watch_hours", "scatter_watch_hours_vs_monthly_fee.png")

    corr_matrix = plot_heatmap(df, NUMERIC_COLUMNS, "numeric_correlation_heatmap.png")
    save_dataframe(corr_matrix.round(4), "correlation_matrix.csv")
    plot_pairplot(df, ["watch_hours", "last_login_days", "monthly_fee", "avg_watch_time_per_day"], "pairplot_key_features.png")
    plot_grouped_boxplot(df, "subscription_type", "watch_hours", "boxplot_watch_hours_by_subscription_type.png")
    plot_grouped_boxplot(df, "favorite_genre", "watch_hours", "boxplot_watch_hours_by_genre.png")
    plot_grouped_boxplot(df, "region", "watch_hours", "boxplot_watch_hours_by_region.png")

    log(report_lines, "Univariate, bivariate, and multivariate plots were saved in outputs/plots.")
    log(report_lines, "Correlation matrix:")
    log(report_lines, corr_matrix.round(3).to_string())

    log_heading(report_lines, "Task 3 - Missing Values and Outliers")
    missing_values = df.isnull().sum()
    missing_df = missing_values.to_frame(name="missing_count")
    save_dataframe(missing_df, "missing_values.csv")
    log(report_lines, "Missing value counts:")
    log(report_lines, missing_values.to_string())
    if missing_values.sum() == 0:
        log(report_lines, "No missing values were found in the raw dataset, so no imputation was applied.")
        log(report_lines, "If missing values existed, numeric columns would be filled using mean or median and categorical columns using mode.")

    outliers_df = outlier_summary(df, ["age", "watch_hours", "last_login_days", "monthly_fee", "number_of_profiles", "avg_watch_time_per_day"])
    save_dataframe(outliers_df, "outlier_summary.csv")
    log(report_lines, "")
    log(report_lines, "Outlier summary using the IQR rule:")
    log(report_lines, outliers_df.round(3).to_string(index=False))
    log(report_lines, "Impact of outliers: extreme engagement values can distort averages, widen variance, and bias linear regression coefficients.")

    log_heading(report_lines, "Task 4 - Spread of Data")
    distribution_df = distribution_statistics(df, ["age", "watch_hours", "last_login_days", "monthly_fee", "number_of_profiles", "avg_watch_time_per_day"])
    save_dataframe(distribution_df.round(4), "distribution_statistics.csv")
    log(report_lines, distribution_df.round(4).to_string())
    log(report_lines, "Interpretation: variables with skewness near 0 are closer to symmetric; larger positive or negative skewness indicates skewed distributions.")
    log(report_lines, "Kurtosis helps identify heavier tails and sharper peaks compared with a normal distribution.")

    log_heading(report_lines, "Task 5 - Automated EDA")
    log(report_lines, "Reusable functions included: plot_histogram_and_boxplot, plot_categorical_count, plot_scatter, plot_heatmap, plot_pairplot, plot_grouped_boxplot, outlier_summary, distribution_statistics.")
    log(report_lines, "The script uses describe(), info(), isnull(), and corr() directly and saves their outputs.")

    log_heading(report_lines, "Task 6 and Task 7 - Regression Analysis")
    log(report_lines, "Dependent variable selected for regression: watch_hours")
    log(report_lines, "Simple linear regression feature: avg_watch_time_per_day")
    log(report_lines, f"Multiple linear regression features: {REGRESSION_FEATURES}")

    covariance_value = df["watch_hours"].cov(df["avg_watch_time_per_day"])
    correlation_value = df["watch_hours"].corr(df["avg_watch_time_per_day"])
    log(report_lines, f"Covariance between watch_hours and avg_watch_time_per_day: {covariance_value:.4f}")
    log(report_lines, f"Correlation between watch_hours and avg_watch_time_per_day: {correlation_value:.4f}")

    train_df, temp_df = train_test_split(df, test_size=0.40, random_state=42, stratify=df["churned"])
    validation_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["churned"])
    log(report_lines, f"Training set size: {train_df.shape}")
    log(report_lines, f"Validation set size: {validation_df.shape}")
    log(report_lines, f"Testing set size: {test_df.shape}")

    y_train_reg = train_df["watch_hours"]
    y_val_reg = validation_df["watch_hours"]
    y_test_reg = test_df["watch_hours"]

    simple_model = LinearRegression()
    simple_model.fit(train_df[["avg_watch_time_per_day"]], y_train_reg)

    simple_results: list[dict[str, float | str]] = []
    for split_name, features, target in [
        ("train", train_df[["avg_watch_time_per_day"]], y_train_reg),
        ("validation", validation_df[["avg_watch_time_per_day"]], y_val_reg),
        ("test", test_df[["avg_watch_time_per_day"]], y_test_reg),
    ]:
        preds = simple_model.predict(features)
        metrics = regression_metrics(target, preds)
        simple_results.append({"model": "simple_linear_regression", "split": split_name, **metrics})

    multi_model = LinearRegression()
    multi_model.fit(train_df[REGRESSION_FEATURES], y_train_reg)
    multi_results: list[dict[str, float | str]] = []
    for split_name, features, target in [
        ("train", train_df[REGRESSION_FEATURES], y_train_reg),
        ("validation", validation_df[REGRESSION_FEATURES], y_val_reg),
        ("test", test_df[REGRESSION_FEATURES], y_test_reg),
    ]:
        preds = multi_model.predict(features)
        metrics = regression_metrics(target, preds)
        multi_results.append({"model": "multiple_linear_regression", "split": split_name, **metrics})

    regression_results_df = pd.DataFrame(simple_results + multi_results)
    save_dataframe(regression_results_df.round(6), "regression_results.csv")
    log(report_lines, "Regression results:")
    log(report_lines, regression_results_df.round(6).to_string(index=False))

    plot_df = test_df[["avg_watch_time_per_day"]].copy()
    plot_df["actual_watch_hours"] = y_test_reg.to_numpy()
    plot_df["predicted_watch_hours"] = simple_model.predict(test_df[["avg_watch_time_per_day"]])
    plot_df = plot_df.sort_values(by="avg_watch_time_per_day")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df, x="avg_watch_time_per_day", y="actual_watch_hours", label="Actual", alpha=0.6)
    plt.plot(plot_df["avg_watch_time_per_day"], plot_df["predicted_watch_hours"], color="red", label="Predicted")
    plt.title("Simple Linear Regression Fit on Test Data")
    plt.xlabel("avg_watch_time_per_day")
    plt.ylabel("watch_hours")
    plt.legend()
    save_current_figure("simple_linear_regression_fit.png")

    log_heading(report_lines, "Task 8 - Overfitting and Underfitting")
    complexity_rows: list[dict[str, float | int]] = []
    for degree in [1, 2, 3, 5]:
        poly_model = Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("linear", LinearRegression()),
            ]
        )
        poly_model.fit(train_df[["avg_watch_time_per_day"]], y_train_reg)
        train_pred = poly_model.predict(train_df[["avg_watch_time_per_day"]])
        test_pred = poly_model.predict(test_df[["avg_watch_time_per_day"]])
        complexity_rows.append(
            {
                "degree": degree,
                "train_mse": mean_squared_error(y_train_reg, train_pred),
                "test_mse": mean_squared_error(y_test_reg, test_pred),
            }
        )
    complexity_df = pd.DataFrame(complexity_rows)
    save_dataframe(complexity_df.round(6), "model_complexity_results.csv")
    log(report_lines, complexity_df.round(6).to_string(index=False))
    log(report_lines, "Underfitting occurs when a model is too simple and performs poorly on both training and testing data.")
    log(report_lines, "Overfitting occurs when training error is very low but testing error increases because the model memorizes training patterns.")

    plt.figure(figsize=(8, 5))
    plt.plot(complexity_df["degree"], complexity_df["train_mse"], marker="o", label="Train MSE")
    plt.plot(complexity_df["degree"], complexity_df["test_mse"], marker="o", label="Test MSE")
    plt.title("Model Complexity vs Error")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    save_current_figure("model_complexity_vs_error.png")

    log_heading(report_lines, "Task 9 - Classification with Logistic Regression")
    y_train_cls = train_df["churned"]
    y_val_cls = validation_df["churned"]
    y_test_cls = test_df["churned"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CLASSIFICATION_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CLASSIFICATION_CATEGORICAL_FEATURES),
        ]
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    logistic_pipeline.fit(train_df[CLASSIFICATION_NUMERIC_FEATURES + CLASSIFICATION_CATEGORICAL_FEATURES], y_train_cls)

    classification_rows: list[dict[str, float | str]] = []
    for split_name, features_df, target in [
        ("train", train_df, y_train_cls),
        ("validation", validation_df, y_val_cls),
        ("test", test_df, y_test_cls),
    ]:
        predictions = logistic_pipeline.predict(features_df[CLASSIFICATION_NUMERIC_FEATURES + CLASSIFICATION_CATEGORICAL_FEATURES])
        metrics = classification_metrics(target, predictions)
        classification_rows.append({"model": "logistic_regression", "split": split_name, **metrics})

        if split_name in {"validation", "test"}:
            matrix = confusion_matrix(target, predictions)
            save_confusion_heatmap(
                matrix,
                f"confusion_matrix_{split_name}.png",
                f"Confusion Matrix ({split_name.title()})",
            )

    classification_df = pd.DataFrame(classification_rows)
    save_dataframe(classification_df.round(6), "classification_results.csv")
    log(report_lines, classification_df.round(6).to_string(index=False))

    test_predictions = logistic_pipeline.predict(test_df[CLASSIFICATION_NUMERIC_FEATURES + CLASSIFICATION_CATEGORICAL_FEATURES])
    test_conf_matrix = confusion_matrix(y_test_cls, test_predictions)
    conf_matrix_df = pd.DataFrame(test_conf_matrix, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
    save_dataframe(conf_matrix_df, "test_confusion_matrix.csv")
    log(report_lines, "Test confusion matrix:")
    log(report_lines, conf_matrix_df.to_string())

    log_heading(report_lines, "Task 10 and Task 11 - Model Evaluation")
    best_regression_test = regression_results_df[regression_results_df["split"] == "test"].sort_values(by="R2", ascending=False)
    best_classification_test = classification_df[classification_df["split"] == "test"]
    log(report_lines, "Regression metrics are reported using MSE, MAE, and R2 score.")
    log(report_lines, best_regression_test.round(6).to_string(index=False))
    log(report_lines, "Classification performance is reported using accuracy and confusion matrix.")
    log(report_lines, best_classification_test.round(6).to_string(index=False))
    log(report_lines, "Comparison: regression estimates engagement hours, while classification predicts churn risk.")

    log_heading(report_lines, "Task 12 - Visualization Summary")
    plot_files = sorted(path.name for path in PLOTS_DIR.glob("*.png"))
    for plot_name in plot_files:
        log(report_lines, f"Generated plot: {plot_name}")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nAnalysis complete. Report saved to: {REPORT_PATH}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
