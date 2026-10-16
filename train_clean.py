"""Clean train-only CV comparison on the UCI Car Evaluation dataset.

This script is an independent, reproducible benchmark alongside the original
assignment notebook; only the selected model is scored on the held-out set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

COLUMNS = ("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
FEATURES = COLUMNS[:-1]
TARGET_MAP = {"unacc": "Unacceptable", "acc": "Acceptable", "good": "Good", "vgood": "Good"}
CATEGORY_ORDER = [
    ["low", "med", "high", "vhigh"],
    ["low", "med", "high", "vhigh"],
    ["2", "3", "4", "5more"],
    ["2", "4", "more"],
    ["small", "med", "big"],
    ["low", "med", "high"],
]
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"


def load_data(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the original UCI categories with the assignment's three labels."""
    source = Path(path) if path else DATA_URL
    df = pd.read_csv(source, header=None, names=COLUMNS, dtype=str)
    if df.empty or df.isna().any().any():
        raise ValueError("Dataset cannot be empty or contain missing categorical fields")
    target = df["class"].map(TARGET_MAP)
    if target.isna().any():
        raise ValueError("Unknown target label in dataset")
    for col, valid in zip(FEATURES, CATEGORY_ORDER):
        if not df[col].isin(valid).all():
            raise ValueError(f"Unexpected input category in {col}")
    return df.loc[:, FEATURES], target


def compare_models(X: pd.DataFrame, y: pd.Series, *, seed: int = 42, quick: bool = False) -> dict:
    if len(X) != len(y) or y.nunique() != 3:
        raise ValueError("Provide matching X/y with all three target classes")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    def pipe(model):
        return Pipeline([
            ("encode", OrdinalEncoder(categories=CATEGORY_ORDER, handle_unknown="use_encoded_value", unknown_value=-1)),
            ("model", model),
        ])

    estimators = {
        "Decision Tree": (
            pipe(DecisionTreeClassifier(random_state=seed)),
            {"model__max_depth": [4, 8] if quick else [3, 5, 8, 12, None]},
        ),
        "Random Forest": (
            pipe(RandomForestClassifier(random_state=seed, n_jobs=1)),
            {"model__n_estimators": [10] if quick else [100, 200],
             "model__max_depth": [None] if quick else [None, 10],
             "model__max_features": ["sqrt"] if quick else ["sqrt", None]},
        ),
        "AdaBoost": (
            pipe(AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),random_state=seed)),
            {"model__n_estimators": [10] if quick else [100, 200],
             "model__learning_rate": [0.1] if quick else [0.1, 1.0]},
        ),
    }
    cv_results = {}
    winner, winning_search, winning_cv = None, None, float("-inf")
    for name, (estimator, grid) in estimators.items():
        search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, refit=True, n_jobs=1)
        search.fit(Xtr, ytr)
        score = float(search.best_score_)
        cv_results[name] = score
        if score > winning_cv:
            winner, winning_search, winning_cv = name, search, score

    # Crucially, the test partition is used once, after CV-based selection.
    predictions = winning_search.best_estimator_.predict(Xte)
    return {
        "selected_model": winner,
        "selection_metric": "3-fold training CV macro-F1",
        "cv_macro_f1": cv_results,
        "best_parameters": winning_search.best_params_,
        "test_macro_f1": float(f1_score(yte, predictions, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(yte, predictions)),
        "test_report": classification_report(yte, predictions, zero_division=0, output_dict=True),
        "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="Optional local car.data; otherwise UCI URL")
    parser.add_argument("--output", type=Path, default=Path("results/new_clean_comparison.json"))
    args = parser.parse_args()
    X, y = load_data(args.data)
    result = compare_models(X, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["selected_model", "cv_macro_f1", "test_macro_f1", "test_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
