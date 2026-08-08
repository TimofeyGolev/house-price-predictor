import pickle
import pandas as pd


# Load trained model
with open("models/house_model.pkl", "rb") as file:
    model = pickle.load(file)


# Get house information from the user
state = input("State: ").strip().title()
city = input("City: ").strip().title()
area_sqft = float(input("Area (sqft): ").strip())
bedrooms = int(input("Bedrooms: ").strip())
bathrooms = int(input("Bathrooms: ").strip())
year_built = int(input("Year built: ").strip())
garage = input("Garage (Yes/No): ").strip().title()
distance_center = float(input("Distance from center (miles): ").strip())
condition = input("Condition (Excellent/Good/Average/Poor): ").strip().title()


# Create a DataFrame for the new house
house = pd.DataFrame([
    {
        "State": state,
        "City": city,
        "Area_sqft": area_sqft,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Year_Built": year_built,
        "Garage": garage,
        "Distance_Center": distance_center,
        "Condition": condition
    }
])


# Predict the house price
prediction = model.predict(house)


# Display the predicted price
print(f"\nPredicted price: ${prediction[0]:,.2f}")
