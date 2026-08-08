## House Price Predictor
A machine learning project that predicts house prices based on property characteristics.


## Project Overview
The model uses information about a house to predict its price.

The current dataset contains 50,000 generated house records.

Features used by the model:

State
City
Area (sqft)
Bedrooms
Bathrooms
Year Built
Garage
Distance from city center
Condition


## Model
The project uses a Random Forest Regressor.

Categorical features are converted using OneHotEncoder, and the preprocessing and model are combined into a single scikit-learn Pipeline.


## Current Results
The current model achieves:

MAE: ~$34,443
R²: 0.9814
The dataset contains random price variation of up to ±$50,000, so the model cannot perfectly predict every generated price.


## Dataset
The dataset is synthetically generated for this project.

The price is generated from several factors, including:

city
area
number of bedrooms and bathrooms
year built
garage
condition
distance from the city center
Random variation is also added to make the prices less deterministic.

This dataset is not intended to represent real-world housing market prices.


## Project Structure
house-price-predictor/
│
├── data/
│   └── houses.csv
│
├── models/
│   └── house_model.pkl
│
├── generate_data.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore


## How to Run
Create and activate a virtual environment:

python -m venv .venv
Activate it on Windows:

.venv\Scripts\activate
Install dependencies:

pip install -r requirements.txt
Generate the dataset:

python generate_data.py
Train the model:

python train.py
The trained model will be saved to:

models/house_model.pkl


## Future Development
The current version is a baseline.

Future versions will focus on using real housing market data and expanding the project from simple price prediction to market analysis.

Possible features include:

real property listings
price per square foot
neighborhood analysis
comparable properties
market trends
rental price estimation
identifying potentially overpriced or underpriced properties

The long-term goal is to build a system that can analyze the housing market rather than only predict the price of an individual house.
