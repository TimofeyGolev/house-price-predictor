import pandas as pd
import joblib


# Load model
model = joblib.load("models/house_price_model.pkl")


# Load data
file_path = "data/HouseTS.csv"
df = pd.read_csv(file_path)


# Reduce zipcode cardinality (same as train.py)
df["zipcode"] = df["zipcode"].astype(str)

top_zipcodes = joblib.load("models/top_zipcodes.pkl")

df["zipcode"] = df["zipcode"].where(
    df["zipcode"].isin(top_zipcodes),
    "other"
)


# Repeat predictions until the user chooses to stop
while True:

# Collect city, zipcode, and year from the user
# (city must match dataset format, e.g. "SF"; zipcode stays as text to match df["zipcode"])
    city_input = input("Enter city (e.g. SF, LA, NY): ")
    zipcode_input = input("Enter zipcode: ")
    year_input = input("Enter year: ")


    if int(year_input) < 2012 or int(year_input) > 2023:
        print("⚠️ The model was trained on data from 2012 to 2023 — predictions outside this range may be less reliable.")

# If the ZIP code isn't common enough (not in top 500), fall back to "other" like in training
    if zipcode_input not in top_zipcodes:
        print("⚠️ This ZIP code is rare in the dataset — prediction accuracy may be lower.")
        zipcode_input = "other"


# Find the most recent record matching the city and (possibly adjusted) zipcode
    filtered = df[(df["city"] == city_input) & (df["zipcode"] == zipcode_input)]
    latest_record = filtered.sort_values("date", ascending=False).head(1)


# Skip this attempt and let the user try again, instead of ending the whole program
    if latest_record.empty:
        print("No matching data found for this city and ZIP code.")
        continue


# Update year to match what the user asked for
    latest_record["year"] = int(year_input)


# Select only the features the model expects
    features = latest_record[
        [
            "city", "zipcode", "year", "median_list_ppsf", "homes_sold",
            "pending_sales", "new_listings", "inventory", "median_dom",
            "avg_sale_to_list", "sold_above_list", "off_market_in_two_weeks",
            "bank", "bus", "hospital", "mall", "park", "restaurant", "school",
            "station", "supermarket", "Total Population", "Median Home Value",
            "Median Age", "Per Capita Income", "Total Families Below Poverty",
            "Total Housing Units", "Median Rent", "Total Labor Force",
            "Unemployed Population", "Total School Age Population",
            "Total School Enrollment", "Median Commute Time"
        ]
    ]


# Predict and show the result
    prediction = model.predict(features)
    print(f"Predicted price: ${prediction[0]:,.0f}")


# Ask if the user wants to predict another house, otherwise exit the loop
    again = input("Predict another house? (yes/no): ")
    if again.lower() !="yes":
        break
