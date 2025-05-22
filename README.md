# 🚗 Car Price Prediction using Python

This project demonstrates how to **predict used car prices** based on features like **fuel type**, **transmission**, **car age**, and more. It uses a **Random Forest Regressor**, a powerful ensemble machine learning algorithm, to model the relationship between car features and selling price.

---

## Project Description

> **Car price prediction** helps estimate the selling price of a used car by analyzing historical data such as fuel type, kilometers driven, and vehicle age.

This project processes raw car data (`car data.csv`), cleans it, and builds a machine learning model that can predict a car's selling price based on its attributes.

---

## Files Included

- `car data.csv` — Raw dataset  
- `cleaning_dat.py` — Script to clean the raw data:  
  - Removes duplicates and outliers  
  - Handles missing values  
  - Converts year to `Car_Age`  
  - Saves the cleaned dataset as `cleaned_car_data.csv`  
- `carPrice_prediction.py` — Machine learning pipeline to:  
  - Encode categorical variables using Target Encoding  
  - Train a Random Forest Regressor  
  - Evaluate using R² and Mean Squared Error  
  - Visualize actual vs. predicted prices

---

### 1. Install Required Packages

Ensure Python 3.x is installed, then run:

```bash
pip install pandas matplotlib seaborn scikit-learn category_encoders
```

### 2. Run the `cleaning_data.py` which will clean the data and then run `carPrice_prediction.py`
