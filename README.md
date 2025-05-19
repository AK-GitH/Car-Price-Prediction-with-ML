# Car Price Prediction

A machine learning project to predict the **selling price of used cars** based on features like fuel type, transmission, car age, and more.

## Overview

This project uses a **Random Forest Regressor** within a scikit-learn pipeline that handles preprocessing of categorical and numerical features automatically. The model is evaluated using **Mean Squared Error (MSE)** and **R² Score**. The project also includes visualizations to provide insights into price trends by fuel type and transmission.

## Files

- `car data.csv` – Original dataset of used car listings.  
- `clean_data.py` – Script to clean and preprocess raw data, removing duplicates, handling missing values, creating derived features, and removing outliers. Saves cleaned data to `cleaned_car_data.csv`.  
- `predict_car_price.py` – Main modeling script: loads cleaned data, trains the model, evaluates, and shows results and visualizations.

## Highlights

- Robust data cleaning pipeline separate from modeling.  
- Automatic feature preprocessing with `ColumnTransformer` and `OneHotEncoder`.  
- Accurate regression model using `RandomForestRegressor`.  
- Clear evaluation metrics: MSE and R².  
- Visual insights:  
  - **Actual vs Predicted Prices**  
  - **Average Selling Price by Fuel Type**  
  - **Average Selling Price by Transmission**

## Installation

Make sure you have Python 3.x installed, then install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn