import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# 1. Load the dataset
df = pd.read_csv("data/titanic.csv")


# 2. Handle missing values

# Fill missing Age values with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Remove Cabin because it contains a large number of missing values
df.drop("Cabin", axis=1, inplace=True)

# Fill missing Embarked values with the most common category
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# 3. Encode categorical features

# One-hot encode Sex
df = pd.get_dummies(df, columns=["Sex"], dtype=int)

# One-hot encode Embarked
df = pd.get_dummies(df, columns=["Embarked"], dtype=int)

# Visualize Fare outliers before removal
sns.boxplot(x=df["Fare"])
plt.title("Fare Distribution Before Outlier Removal")
plt.show()


# 4. Detect and remove Fare outliers using IQR

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove rows with Fare above the upper IQR boundary
df = df[df["Fare"] <= upper_bound]


# 5. Standardize numerical features

scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(
    df[["Age", "Fare"]]
)


# 6. Save the cleaned dataset

df.to_csv("data/titanic_cleaned.csv", index=False)


# 7. Final summary

print("--- Final Dataset Summary ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())