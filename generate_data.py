import pandas as pd
import random

# Store all generated houses
houses = []

# Generate 1500 random houses
for i in range(1500):
    #Generate house features
    area = random.randint(30, 150)
    rooms = random.randint(1, 6)
    floor = random.randint(1, 20)

    # Calculate house price
    price = (
        area * 3000
        + rooms * 20000
        + floor * 3000
        + random.randint(-10000, 10000)
    )

    # Add the house to the dataset
    houses.append([
        area,
        rooms,
        floor,
        price
    ])

# Create a DataFrame
df = pd.DataFrame(
    houses,
    columns=["Area", "Rooms", "Floor", "Price"]
)

# Save the dataset as a CSV file
df.to_csv("data/houses.csv", index=False)

# Display dataset information
print("Dataset created!")

# Display the first five rows
print(df.head())
# Display the dataset size(number of rows and columns)
print(df.shape)
