# ML-5
# ML Task 5 — Data Cleaning & Feature Selection

This repository contains two notebooks that work through an initial data-preparation pass on a bank customer churn dataset, followed by a simple feature-selection / linear regression demo.

## Contents

| File | Description |
|---|---|
| `Churn_Modelling (MAM) - Churn_Modelling (3).csv` | Raw dataset — 10,000 bank customer records with 14 columns. |
| `ML_Task_5(dc).ipynb` | **Data Cleaning** notebook — inspects and handles missing values in the dataset. |
| `ML_Task_5(fs).ipynb` | **Feature Selection** notebook — demonstrates a basic train/test split and linear regression workflow. |

## Dataset

The dataset (`Churn_Modelling.csv`) contains the following columns:

- `RowNumber`, `CustomerId`, `Surname` — identifiers
- `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary` — customer attributes
- `Exited` — target variable (1 = customer churned, 0 = customer retained)

## 1. Data Cleaning (`ML_Task_5(dc).ipynb`)

Steps performed:

1. **Load the data** with `pandas.read_csv`.
2. **Inspect structure and nulls** using `df.info()` and `df.isnull().sum()`.
   - Missing values found in:
     - `Gender` — 47 missing
     - `Age` — 66 missing
3. **Handle missing data**:
   - Explored dropping all columns containing nulls (`df.dropna(axis=1)`) to see the resulting shape.
   - Computed the mean (~38.92) and median (37.0) of `Age`.
   - Chose to **impute missing `Age` values with the column mean** rather than dropping data, preserving all 10,000 rows and 14 columns.
4. Final cleaned dataset has no missing values in `Age` (mean-imputed); `Gender` nulls remain to be addressed in a follow-up step.

## 2. Feature Selection (`ML_Task_5(fs).ipynb`)

A minimal end-to-end regression example used to illustrate the train/test split and modeling workflow:

1. Creates a small synthetic dataset (`X`, `y`).
2. Splits it into training and test sets with `train_test_split` (70/30 split, `random_state=42`).
3. Fits a `LinearRegression` model from `scikit-learn`.
4. Generates predictions on the test set and compares them against actual values.

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

1. Clone the repository.
2. Place `Churn_Modelling (MAM) - Churn_Modelling (3).csv` in the same directory as the notebooks (or update the file path in the first cell).
3. Open and run `ML_Task_5(dc).ipynb` to clean the data.
4. Open and run `ML_Task_5(fs).ipynb` to see the feature selection / regression demo.

## Notes / Next Steps

- `Gender` still contains 47 missing values — consider imputing with the mode or a separate "Unknown" category.
- The feature selection notebook currently uses toy/synthetic data rather than the churn dataset itself — a natural next step is applying feature selection (e.g., correlation analysis, `SelectKBest`) directly to the cleaned churn data ahead of building a classification model for `Exited`.
