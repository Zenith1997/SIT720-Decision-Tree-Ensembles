# Decision Trees vs Ensembles: Car Evaluation

**A comparison of a simple decision tree, Random Forest and AdaBoost on an imbalanced, three-class categorical dataset.**

I wanted to see how much a tree ensemble improves upon a single interpretable decision tree — and how model depth changes performance. One lesson was that it's easy to make a comparison look convincing by plotting test results repeatedly. A cleaner evaluation chooses everything using cross-validation on training data, and only checks the test partition after making that choice.

## Original notebook results

The saved assignment notebook reports Random Forest **training CV macro-F1 0.9565** and **held-out test macro-F1 0.9725**. The selected Decision Tree also scored **0.9725** on that test set, so the evidence does not support claiming Random Forest was unambiguously better.

**Important qualification:** The notebook explores test-set performance for multiple tree depths and prints the best model according to test macro-F1. That is useful for learning, but it is not a strictly untouched-test selection protocol. The standalone `train_clean.py` is provided to correct that methodological issue for new runs. No replacement UCI metrics are claimed until it is executed on the actual dataset.

## Explore

- [Original executed notebook](SIT720_9.1P_Corrected_Executed.ipynb)
- [Coursework report](SIT720_9.1P_Corrected_Report.pdf)
- [Training-only CV comparison script](train_clean.py)
- [UCI data source and target-class mapping](DATA.md)
- [Automated tests](tests/test_clean_evaluation.py)

## Reproduce the cleaner evaluation

Python 3.10+:

```bash
python -m pip install -r requirements.txt
python train_clean.py
```

The script downloads `car.data` from UCI unless you provide `--data car.data`, uses a stratified 80:20 split, selects model and hyperparameters with **3-fold training-only CV macro-F1**, then evaluates the selected model on the held-out test set **once**. It writes new findings to `results/new_clean_comparison.json`, separate from the historical report.

Run the tests without the UCI dataset:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

For the full original notebook with figures: install `requirements-notebook.txt`. The original notebook and report were not edited or rerun during this repository refactor.

**Author:** Zenith Anthony Dabarera | [GitHub](https://github.com/Zenith1997)
