# Task 1: Data Cleaning & Preprocessing

## Objective

Clean and preprocess the Titanic dataset for machine learning.
## Dataset

Titanic Dataset

- Original rows: 891
- Original columns: 12
- Final rows: 775
- Final columns: 14

## Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Data Cleaning & Preprocessing Steps

### 1. Dataset Exploration

The dataset was explored to understand:

- Number of rows and columns
- Column names
- Data types
- Missing values
- Duplicate rows

### 2. Handling Missing Values

Missing values were found in:

- `Age`
- `Cabin`
- `Embarked`

#### Age

The missing `Age` values were replaced using the median.

- Mean Age: 29
- Median Age: 28

The median was used because it is less affected by extreme values.

#### Cabin

The `Cabin` column contained 687 missing values. Since a large amount of information was missing, the column was removed.

#### Embarked

The `Embarked` column had 2 missing values.

The category counts were:

- S: 644
- C: 168
- Q: 77

The missing values were replaced with the most common category, `S`.

### 3. Categorical Encoding

Categorical features were converted into numerical features.

One-hot encoding was applied to:

- `Sex`
- `Embarked`

This created the following numerical columns:

- `Sex_female`
- `Sex_male`
- `Embarked_C`
- `Embarked_Q`
- `Embarked_S`

### 4. Outlier Detection

Outliers were detected using a boxplot and the IQR (Interquartile Range) method.

For the original `Fare` values:

- Q1: 7.9
- Q3: 31
- IQR: 23.1
- Upper bound: 65.6344

Fare values above the upper bound were treated as potential outliers.

A total of 116 rows were removed.

The dataset was reduced from 891 rows to 775 rows.

### 5. Feature Scaling

After handling the outliers, `Age` and `Fare` were standardized using `StandardScaler`.

Standardization transforms numerical features to a common scale centered around 0.

## Final Dataset

The final processed dataset contains:

- 775 rows
- 14 columns
- No missing values
- No duplicate rows
- Numerical encoding for categorical features
- Standardized `Age` and `Fare`

## Files

```text
Task-1-Data-Cleaning/
│
├── data/
│   ├── titanic.csv
│   └── titanic_cleaned.csv
│
├── task1_data_cleaning.py
│
└── README.md