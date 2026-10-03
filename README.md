# House Price Predictor

A machine learning project that predicts real estate market prices for a city, ZIP code, and year using real historical market data.

## About the project

I built this project to learn how machine learning works in a real project — from working with real data to training and testing a model.

The dataset (HouseTS) was taken from the internet. It has real estate market data for about 30 US cities from 2012 to 2023. It includes city, ZIP code, year, and other information like population, income, schools, and more.

The goal is to predict an approximate market price using this data.

## Important: What this model actually predicts

This model does **not** predict the price of one specific house using features like square footage, bedrooms, or bathrooms. The dataset does not have this type of information.

Instead, each row is a market snapshot for one ZIP code in one month. Because of this, the model predicts a **market-level price** — an estimate of the general price level in a city or ZIP code at a certain time.

## Tech stack

* Python
* pandas
* scikit-learn

  * RandomForestRegressor
  * Pipeline
  * ColumnTransformer
  * OneHotEncoder
* joblib — to save and load the model

## Project structure

```text
house-price-predictor/
│
├── data/
│   └── HouseTS.csv
│
├── src/
│   ├── explore_housets.py   # data exploration (EDA)
│   ├── train.py             # trains and saves the model
│   └── predict.py           # makes predictions
│
├── models/
│   ├── house_price_model.pkl
│   └── top_zipcodes.pkl
│
└── README.md
```

## How it works

### 1. Explore the data (`explore_housets.py`)

I checked the price distribution, correlations, and some other parts of the dataset. I also found some bad or suspicious rows.

### 2. Train the model (`train.py`)

* Load 200,000 rows from the dataset using a random sample.
* Split the data by time:

  * **2012–2021** → training
  * **2022–2023** → testing
* I used a time-based split instead of a random split because this is time series data. The model should learn from the past and then predict the future.
* Reduce the number of ZIP code categories (see the section below).
* Train a Random Forest model inside a Pipeline.
* Save the trained model and the ZIP code list.

### 3. Make predictions (`predict.py`)

* Load the saved model.
* Ask the user for a city, ZIP code, and year.
* Find the latest known data for that city and ZIP code.
* Use this data for the other features.
* Predict the price and print the result.
* The user can make another prediction or stop the program.

## Results

Final model: **200,000 rows, Random Forest with 100 trees**

| Metric |    Value |
| ------ | -------: |
| MAE    |  $62,169 |
| RMSE   | $122,311 |
| R²     |    0.918 |

## The ZIP code problem

This was the biggest problem I had in the project.

The dataset has **6,226 unique ZIP codes**. When I used all of them with OneHotEncoder, the table became very large. My computer has 8GB of RAM, and it could not handle it. Training froze for hours. One time, I even left it overnight, but it still did not finish.

### First fix

I kept only the **500 most common ZIP codes** in the whole dataset. All other ZIP codes were grouped into one category called `"other"`.

But then I found another problem.

The top-500 list was not fair between cities. Some cities had many more rows in the dataset, so they took most of the 500 places.

For example:

* Boston: 211 ZIP codes
* Atlanta: 203 ZIP codes
* Los Angeles, San Francisco, and New York: only 1 ZIP code each

This meant the model could not work well for many cities.

### Real fix

I changed the logic to select the **top 20 ZIP codes for each city separately**, instead of selecting the top 500 ZIP codes for the whole dataset.

This gave every city a similar number of ZIP codes.

It also improved the model:

**R²: 0.898 → 0.918**

This was the best result I got in the project.

I also found another bug. `train.py` and `predict.py` were calculating the top ZIP codes in different ways. Because of this, a ZIP code that the model knew could sometimes appear as `"unknown"` in `predict.py`.

I fixed this by saving the exact ZIP code list used during training in `top_zipcodes.pkl`. Now `predict.py` uses the same list.

## Feature importance and a possible data leakage problem

I checked which features were the most important for the model.

**Median Home Value** had about **69% of the feature importance**. This was much higher than the other features.

This made me think that maybe this feature is too close to the real target price. The model could be using this one number instead of really learning the relationship between the features and the price.

I tested this by removing Median Home Value and training the model again.

The result was interesting: another feature, **median_list_ppsf**, became the most important feature with about **70% importance**.

R² only dropped a little:

**0.918 → 0.881**

This makes me think that the dataset has several features that are already closely related to the target price.

This is a limitation of the dataset that I want to understand better in the future.

## Error analysis

The biggest prediction errors happen in **San Francisco, New York, and Miami**.

These cities have very different prices in different areas of the same city. There can be very expensive areas next to much cheaper areas, so the model has a harder time making accurate predictions.

Most of the biggest errors also happened on ZIP codes marked as `"other"`.

This is another sign that the ZIP code grouping is one of the main sources of error.

## Known limitations

* The model predicts a market-level price, not the price of one specific house.
* ZIP codes outside each city's top 20 are grouped into `"other"`.
* Some features may be too closely related to the target price. I have not fully solved this yet.
* Predictions for years outside 2012–2023 are less reliable because the model has not seen this data.
* The model is less accurate for very expensive and rare properties because there are not many examples of them in the dataset.

## Next steps

* Try target encoding for ZIP codes. Instead of treating ZIP codes as categories, I could use a number such as the average historical price. This could help with rare ZIP codes without creating too many categories.
* Keep improving the model based on error analysis.
* Investigate the possible data leakage problem more deeply.

## How to run

```bash
# Train the model
python src/train.py

# Make predictions
python src/predict.py
```

