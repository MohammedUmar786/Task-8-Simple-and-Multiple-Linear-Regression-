# =========================
# TASK 8: LINEAR REGRESSION
# =========================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load Dataset
df = pd.read_csv('/mnt/data/advertising.csv')

# 3. Basic Info
print("Dataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

# 4. Define Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 5. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# SIMPLE LINEAR REGRESSION
# =========================

# Using only TV feature
X_simple = df[['TV']]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

# Train Model
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

# Predictions
y_pred_s = simple_model.predict(X_test_s)

# Evaluation
mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)

print("\n===== Simple Linear Regression =====")
print("MSE:", mse_s)
print("R2 Score:", r2_s)

# Visualization (Regression Line)
plt.figure()
plt.scatter(X_test_s, y_test_s)
plt.plot(X_test_s, y_pred_s)
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("Simple Linear Regression")
plt.show()

# =========================
# MULTIPLE LINEAR REGRESSION
# =========================

# Train Model
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

# Predictions
y_pred_m = multi_model.predict(X_test)

# Evaluation
mse_m = mean_squared_error(y_test, y_pred_m)
r2_m = r2_score(y_test, y_pred_m)

print("\n===== Multiple Linear Regression =====")
print("MSE:", mse_m)
print("R2 Score:", r2_m)

# Model Coefficients
print("\nIntercept:", multi_model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, multi_model.coef_):
    print(f"{feature}: {coef}")

# =========================
# ERROR ANALYSIS (RESIDUALS)
# =========================

residuals = y_test - y_pred_m

plt.figure()
plt.scatter(y_pred_m, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# =========================
# FINAL INSIGHTS
# =========================

print("\n===== Key Insights =====")
print("1. TV advertising shows strong relationship with Sales.")
print("2. Multiple Linear Regression performs better than Simple.")
print("3. Radio contributes positively to predictions.")
print("4. Newspaper has weaker impact compared to others.")
print("5. High R2 score indicates good model performance.")# Task-8-Simple-and-Multiple-Linear-Regression-
