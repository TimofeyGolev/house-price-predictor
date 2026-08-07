import joblib
import pandas as pd

# Download the trained model
model = joblib.load("models/house_model.pkl")

# Get user input
area = int(input("Enter area:"))
rooms = int(input("Enter number of rooms:"))
floor = int(input("Enter floor:"))

#Create input data for the model
input_data = pd.DataFrame(
    [[area, rooms, floor]],
    columns=["Area", "Rooms", "Floor"]
)


# Make a price prediction
prediction = model.predict(input_data)

# Get the predicted value
prediction = prediction[0]

# Display prediction
print(f"Predicted price: ${prediction:,.0f}")
