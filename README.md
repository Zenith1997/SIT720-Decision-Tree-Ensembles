# Decision Trees and Ensemble Classification

Compare interpretable trees, Random Forest and AdaBoost on imbalanced car-evaluation classes.

**Python · scikit-learn · Decision Trees · Random Forest · AdaBoost**

## Project at a glance

The saved Random Forest experiment reports CV macro F1 of 0.9565 and test macro F1 of 0.9725. AdaBoost reports test macro F1 of 0.8348. The tuned Decision Tree also reports 0.9725 on the test set.

## Technical work

- Categorical encoding and class-distribution inspection
- Tree visualization and root-split interpretation
- Depth and bias–variance analysis
- Ensemble tuning, confusion matrices and macro F1 comparison

## Explore the evidence

- [SIT720_9.1P_Corrected_Executed.ipynb](SIT720_9.1P_Corrected_Executed.ipynb)
- [SIT720_9.1P_Corrected_Report.pdf](SIT720_9.1P_Corrected_Report.pdf)

## Run locally

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install jupyter numpy pandas matplotlib scikit-learn
jupyter notebook
```

The notebook loads a local car.data file when present, otherwise its configured UCI URL.

Open the linked notebook and run the cells in order. Package names reflect the source imports; a fresh environment was not tested during this documentation update. Dataset downloads require internet access.

## Limitations and status

Uses the UCI Car Evaluation data with three grouped classes. The notebook plots repeated test-set comparisons; its results should not be presented as a single untouched-test benchmark.

The repository preserves saved notebook outputs. Results are from those outputs, not a new execution.

**Author:** Zenith Anthony Dabarera · SIT720 Machine Learning  
[AI/ML portfolio](https://github.com/Zenith1997)
