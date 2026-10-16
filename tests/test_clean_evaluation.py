from pathlib import Path

import pandas as pd
import pytest

from train_clean import CATEGORY_ORDER, FEATURES, load_data, compare_models


def test_target_mapping_and_merged_good_class(tmp_path: Path):
    rows = [
        ["low", "low", "2", "2", "small", "low", "unacc"],
        ["med", "med", "3", "4", "med", "med", "acc"],
        ["high", "high", "4", "more", "big", "high", "good"],
        ["vhigh", "vhigh", "5more", "2", "med", "low", "vgood"],
    ]
    f = tmp_path / "car.data"
    pd.DataFrame(rows).to_csv(f, header=False, index=False)
    X, y = load_data(f)
    assert list(X.columns) == list(FEATURES)
    assert y.to_list() == ["Unacceptable", "Acceptable", "Good", "Good"]


def test_unknown_category_rejected(tmp_path: Path):
    f = tmp_path / "car.data"
    f.write_text("low,low,2,2,giant,low,unacc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected input category"):
        load_data(f)


def test_train_only_cv_selects_and_reports_final_test():
    # Synthetic categories only to test script mechanics, not claimed UCI accuracy.
    values = []
    for i in range(120):
        cat = [CATEGORY_ORDER[k][(i // (2 ** k)) % len(CATEGORY_ORDER[k])] for k in range(6)]
        values.append(cat)
    X = pd.DataFrame(values, columns=FEATURES)
    y = pd.Series(["Unacceptable", "Acceptable", "Good"] * 40)
    result = compare_models(X, y, quick=True)
    assert set(result["cv_macro_f1"]) == {"Decision Tree", "Random Forest", "AdaBoost"}
    assert result["selected_model"] in result["cv_macro_f1"]
    assert result["train_rows"] + result["test_rows"] == 120
    assert 0 <= result["test_macro_f1"] <= 1
