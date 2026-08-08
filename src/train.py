import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
df = pd.read_csv("data/houses.csv")


# Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"]


# Identify columns with text data
categorical_features = [
    "State",
    "City",
    "Garage",
    "Condition"
]


# Identify columns with numerical data
numerical_features = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Year_Built",
    "Distance_Center"
]


# Create preprocessing for text columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            "Juice",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Create machine learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train the model
model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("MAE:", mae)
print("R²:", r2)


# Save trained model
with open("models/house_model.pkl", "wb") as file:
    pickle.dump(model, file)


print("Model saved successfully")
