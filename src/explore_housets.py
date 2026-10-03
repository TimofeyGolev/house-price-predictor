import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


file_path = "data/HouseTS.csv"

df = pd.read_csv(file_path)


target = df["price"]

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
        "Median Age",
        "Per Capita Income",
        "Total Families Below Poverty",
        "Total Housing Units",
        "Median Rent",
        "Median Home Value",
        "Total Labor Force",
        "Unemployed Population",
        "Total School Age Population",
        "Total School Enrollment",
        "Median Commute Time"
    ]
]


print("\nFeatures shape:")
print(features.shape)

print("\nTarget shape:")
print(target.shape)


categorical_features = ["city", "zipcode"]

numeric_features = [
    col for col in features.columns
    if col not in categorical_features
]

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


print("\nCategorical features:")
print(categorical_features)

print("\nNumeric features:")
print(numeric_features)
