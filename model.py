import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("data.csv")

# Replace missing values with previous/next available values
data = data.ffill().bfill()

# Input features and output
X = data.drop("Trip_Price", axis=1)
y = pd.to_numeric(data["Trip_Price"], errors="coerce")

# Remove rows where the target is still missing
valid_rows = y.notna()
X = X[valid_rows]
y = y[valid_rows]

# Identify categorical and numerical columns
categorical = X.select_dtypes(include=["object"]).columns
numerical = X.select_dtypes(exclude=["object"]).columns

# Encode categorical values
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numerical", "passthrough", numerical)
])

# Create model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# Train and evaluate
if len(X) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Mean Absolute Error:", round(
        mean_absolute_error(y_test, predictions), 2
    ))

    if y_test.nunique() > 1:
        print("R2 Score:", round(r2_score(y_test, predictions), 4))
    else:
        print("Small dataset: evaluation skipped.")

# Train using all available rows
model.fit(X, y)
joblib.dump(model, "trip_price_model.pkl")

print("Trip price model trained and saved successfully!")
print("Training records:", len(X))