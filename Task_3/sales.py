# ==========================================
# TASK 4: SALES PREDICTION USING PYTHON
# ==========================================

# Step 1: Import Libraries
print("THIS IS THE LATEST VERSION OF sales.py")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv("advertising.csv")

# Step 3: Display Dataset
print("First 5 Rows of Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())
print("Reached Step 5")
# Step 4: Data Visualization

# TV vs Sales
plt.figure(figsize=(6,4))
plt.scatter(df["TV"], df["Sales"])
plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertisement Budget")
plt.ylabel("Sales")
plt.savefig("tv_vs_sales.png")
plt.close()

# Radio vs Sales
plt.figure(figsize=(6,4))
plt.scatter(df["Radio"], df["Sales"])
plt.title("Radio Advertising vs Sales")
plt.xlabel("Radio Advertisement Budget")
plt.ylabel("Sales")
plt.savefig("radio_vs_sales.png")
plt.close()

# Newspaper vs Sales
plt.figure(figsize=(6,4))
plt.scatter(df["Newspaper"], df["Sales"])
plt.title("Newspaper Advertising vs Sales")
plt.xlabel("Newspaper Advertisement Budget")
plt.ylabel("Sales")
plt.savefig("newspaper_vs_sales.png")
plt.close()

# Step 5
print("Reached Step 5")
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]
print("Reached Step 6")

# Step 6
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Reached Step 7")

# Step 7: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

print("Reached Step 8")

# Step 8: Predict Sales
y_pred = model.predict(X_test)

print("Reached Step 9")

# Step 9: Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== Model Evaluation ==========")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)
print("Reached Step 10")
# Step 10: Compare Actual vs Predicted Values
comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nActual vs Predicted Sales:")
print(comparison.head(10))
print("Reached Step 11")
# Step 11: Plot Actual vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs Predicted Sales")
plt.grid(True)
plt.savefig("plot.png")
plt.close()
print("Reached Step 12")
# Step 12: Predict Sales for New Advertisement Budget
print("\n========== Predict New Sales ==========")

tv = float(input("Enter TV Advertisement Budget: "))
radio = float(input("Enter Radio Advertisement Budget: "))
newspaper = float(input("Enter Newspaper Advertisement Budget: "))

new_data = [[tv, radio, newspaper]]

prediction = model.predict(new_data)

print("\nPredicted Sales =", round(prediction[0], 2))

# Step 13: Display Model Equation
print("\n========== Model Details ==========")
print("Intercept:", model.intercept_)

print("\nCoefficients:")
print("TV:", model.coef_[0])
print("Radio:", model.coef_[1])
print("Newspaper:", model.coef_[2])

print("\nModel Equation:")
print(
    f"Sales = {model.intercept_:.2f} + "
    f"({model.coef_[0]:.4f} × TV) + "
    f"({model.coef_[1]:.4f} × Radio) + "
    f"({model.coef_[2]:.4f} × Newspaper)"
)