import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("data/houses.csv")

# Features and target
X = df[['Area', 'Rooms', 'Floor']]
y = df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the machine learning model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "models/house_model.pkl")

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Print evaluation metrics
print("\nMAE:")
print(mae)

print("\nR²:")
print(r2)
