import pandas as pd
import random


# Number of houses to generate
num_houses = 50000


# Cities with approximate base house prices
cities = {
    "Los Angeles": 800000,
    "New York": 900000,
    "Austin": 500000,
    "Miami": 700000,
    "Chicago": 450000,
    "Seattle": 750000,
    "Boston": 800000,
    "Denver": 600000,
    "Dallas": 450000,
    "San Francisco": 1200000
}


# Mapping cities to states
states = {
    "Los Angeles": "California",
    "New York": "New York",
    "Austin": "Texas",
    "Miami": "Florida",
    "Chicago": "Illinois",
    "Seattle": "Washington",
    "Boston": "Massachusetts",
    "Denver": "Colorado",
    "Dallas": "Texas",
    "San Francisco": "California"
}


# Possible house conditions
conditions = [
    "Poor",
    "Average",
    "Good",
    "Excellent"
]


# Empty list for storing generated houses
houses = []


# Generate houses
for _ in range(num_houses):

    # Select random city
    city = random.choice(list(cities.keys()))

    # Get corresponding state
    state = states[city]

    # Generate house features
    area = random.randint(800, 4000)
    bedrooms = random.randint(1, 6)
    bathrooms = random.randint(1, 5)
    year_built = random.randint(1950, 2026)
    garage = random.choice(["Yes", "No"])

    # Distance from city center in miles
    # Random decimal number between 0.5 and 30
    distance_center = round(random.uniform(0.5, 30), 1)

    # Select house condition
    condition = random.choice(conditions)


    #  Start price based on city
    price = cities[city]


    # Larger area increase price
    price += area * 150

    # More bedrooms increase price
    price += bedrooms * 30000

    # More bathrooms increase price
    price += bathrooms * 25000

    # Newer houses usually cost more
    price += (year_built - 1950) * 1000

    # Garage increases property value
    if garage == "Yes":
        price += 40000

    # Adjust price based on condition
    if condition == "Excellent":
        price *= 1.15
    elif condition == "Good":
        price *= 1.05
    elif condition == "Average":
        price *= 1.00
    elif condition == "Poor":
        price *= 0.85


    # Houses farther from center are usually cheaper
    price -= distance_center * 5000

    # Add random variation to make prices more realistic
    price += random.randint(-50000, 50000)


    # Add generated house to the dataset
    houses.append([
        state,
        city,
        area,
        bedrooms,
        bathrooms,
        year_built,
        garage,
        distance_center,
        condition,
        round(price)
    ])


# Create a DataFrame from generated data
df = pd.DataFrame(
    houses,
    columns=[
        "State",
        "City",
        "Area_sqft",
        "Bedrooms",
        "Bathrooms",
        "Year_Built",
        "Garage",
        "Distance_Center",
        "Condition",
        "Price"
    ]
)


# Save dataset as CSV file
df.to_csv("data/houses.csv", index=False)


# Display  dataset size
print("Dataset created:", df.shape)
