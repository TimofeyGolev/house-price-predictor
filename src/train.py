import pandas as pd
import joblib


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)


# Load data
file_path = "data/HouseTS.csv"
df = pd.read_csv(file_path)
df = df.sample(n=200000, random_state=42)

# Date
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Reduce zipcode cardinality
df["zipcode"] = df["zipcode"].astype(str)

top_zipcodes = (
    df.groupby("city")["zipcode"]
    .value_counts()
    .groupby("city")
    .head(20)
    .index.get_level_values("zipcode")
)

# Save the top zipcodes list
joblib.dump(top_zipcodes, "models/top_zipcodes.pkl")

df["zipcode"] = df["zipcode"].where(
    df["zipcode"].isin(top_zipcodes),
    "other"
)

# Target
target = df["price"]

# Features
features = df[
    [
        "city",
        "zipcode",
        "year",
        "median_list_ppsf",
        "homes_sold",
        "pending_sales",
        "new_listings",
        "inventory",
        "median_dom",
        "avg_sale_to_list",
        "sold_above_list",
        "off_market_in_two_weeks",
        "bank",
        "bus",
        "hospital",
        "mall",
        "park",
        "restaurant",
        "school",
        "station",
        "supermarket",
        "Total Population",
        "Median Home Value",
        "Median Age",
        "Per Capita Income",
        "Total Families Below Poverty",
        "Total Housing Units",
        "Median Rent",
        "Total Labor Force",
        "Unemployed Population",
        "Total School Age Population",
        "Total School Enrollment",
        "Median Commute Time"
    ]
]


# Features types
categorical_features = ["city", "zipcode"]

numeric_features = [
    col for col in features.columns
    if col not in categorical_features
]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "Juice",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "WRLD",
            "passthrough",
            numeric_features
        )
    ]
)


# Time split
split_date = "2022-01-01"

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date


X_train = features[train_mask]
X_test = features[test_mask]

y_train = target[train_mask]
y_test = target[test_mask]


# Pipeline + Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    ))
])


# Train
model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print(f"MAE:   ${mae:,.0f}")
print(f"RMSE:  ${rmse:,.0f}")
print(f"R2:    {r2:.4f}")


# Error analysis
errors = predictions - y_test
abs_error = errors.abs()

results = pd.DataFrame({
    "city": X_test["city"],
    "real_price": y_test,
    "predicted_price": predictions,
    "abs_error": abs_error,
    "zipcode": X_test["zipcode"],
})

print(results.sort_values(by="abs_error", ascending=False).head(10))


# Feature importance
importances = model.named_steps["regressor"].feature_importances_
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

print(importance_df.sort_values(by="importance", ascending=False).head(20))


# Save Model
joblib.dump(
    model,
    "models/house_price_model.pkl"
)

print("Model saved.")
