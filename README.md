# Predicting Loan Repayment Ability by Using Machine Learning Methods

Machine learning models for predicting loan repayment ability on the
[Home Credit Default Risk][kaggle] dataset.
Based on the CS229 project [*Loanliness: Predicting Loan Repayment Ability by
Using Machine Learning Methods*][cs229-loanliness].

## Usage

The project is a pipeline of three Jupyter notebooks, run in order:

1. [`01_data_preprocessing.ipynb`](01_data_preprocessing.ipynb) - aggregates
   the auxiliary tables (bureau, previous applications, ...), merges them with
   the main applications table, splits the data and handles encoding, outliers
   and missing values. Writes `dataset/train_set.csv` and
   `dataset/test_set.csv`.
2. [`02_model_training.ipynb`](02_model_training.ipynb) - trains logistic
   regression and XGBoost (tuned with grid-search cross-validation) and a
   fully-connected neural network. Saves the trained models to `models/`.
3. [`03_model_evaluation.ipynb`](03_model_evaluation.ipynb) - loads the models
   from `models/`, compares them on the test set (classification reports, ROC
   curves, feature importance) and draws a conclusion.

Directory structure:

```
dataset/   Home Credit CSV tables + train/test sets are generated here
models/    trained models, saved by notebook 02 and loaded by notebook 03
utils.py   feature-aggregation helpers used by notebook 01
model.py   neural-network class shared by notebooks 02 and 03
```

Before running, download the [Home Credit Default Risk][kaggle] CSV files into
`dataset/`. The `NUM_ROWS` constant in notebook 01 limits how many rows of each
table are used.
Set it to `None` to use all data.

## Running the project

Python 3.10 or newer is required, with the packages:

- *NumPy*
- *Pandas*
- *Matplotlib*
- *scikit-learn*
- *XGBoost*
- *PyTorch*
- *Jupyter*

Required packages can be installed via *pip* with the command:
```sh
pip install numpy pandas matplotlib scikit-learn xgboost torch jupyter
```

Working in a virtual environment is recommended. Start with:

```sh
jupyter notebook 01_data_preprocessing.ipynb
```

## Team members

* Petar Milosavljević, 1108/2025 ([KorsicjeKlosar](https://github.com/KorsicjeKlosar))
* Stefan Milenković, 1076/2024 ([stemil01](https://github.com/stemil01))

[kaggle]: https://www.kaggle.com/competitions/home-credit-default-risk/data
[cs229-loanliness]: https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26644913.pdf
