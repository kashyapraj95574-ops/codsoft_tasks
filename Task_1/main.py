import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Select required columns
data = data[["Survived", "Pclass", "Sex", "Age", "Fare"]]

# Fill missing Age values
data["Age"] = data["Age"].fillna(data["Age"].mean())

# Convert male/female into numbers
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# Features (input)
X = data[["Pclass", "Sex", "Age", "Fare"]]

# Target (output)
y = data["Survived"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Machine Learning model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict survival
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Display result
print("===================================")
print(" TITANIC SURVIVAL PREDICTION")
print("===================================")
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("===================================")