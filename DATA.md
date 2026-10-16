# Data source and class definitions

The original notebook uses the **UCI Machine Learning Repository's Car Evaluation dataset**. The data consists of categorical car evaluation attributes. `car.data` can be obtained from [the original UCI source](https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data), or the new training script can fetch the same URL if it is reachable.

Features: `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`.

The original four UCI labels are grouped into **three classes**, matching the coursework notebook:

- `unacc` => `Unacceptable`
- `acc` => `Acceptable`
- `good` and `vgood` => `Good`

This grouping is important for interpreting all macro-F1 results. The repository doesn't bundle the original dataset; the code downloads or reads it locally. Check the source licence before republishing source data.

The historical notebook plots *test* results across many tree depths, while the new `train_clean.py` selects model and hyperparameters using **training-set CV only**, and scores its selected winner on the test partition once. These are distinct experiments and should not be compared as though they used identical model-selection protocols.
