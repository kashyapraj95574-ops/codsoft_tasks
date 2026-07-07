import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

print("First 5 rows:")
print(df.head())

# Remove missing values
df = df.dropna()

# Remove unnecessary spaces in column names
df.columns = df.columns.str.strip()

# Encode categorical columns
encoder = LabelEncoder()

categorical_columns = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]

for col in categorical_columns:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col].astype(str))

# Convert Year to numeric
if "Year" in df.columns:
    df["Year"] = df["Year"].astype(str).str.extract(r"(\d+)")[0]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Convert Duration to numeric
if "Duration" in df.columns:
    df["Duration"] = df["Duration"].astype(str).str.extract(r"(\d+)")[0]
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
    # Convert Votes to numeric
if "Votes" in df.columns:
    df["Votes"] = df["Votes"].astype(str).str.replace(",", "", regex=False)
    print(df.columns)
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
# Drop rows with missing values again
df = df.dropna()

# Features and target
X = df.drop(["Rating", "Name"], axis=1, errors="ignore")
y = df["Rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Performance")
print("---------------------------")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))