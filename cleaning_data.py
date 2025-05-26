import pandas as pd
import numpy as np  # For log transformation

# Load dataset
df = pd.read_csv("car data.csv")

# Drop Car_Name column
df = df.drop(columns=["Car_Name"], errors='ignore')

# Remove duplicates and reset index
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Define categorical and numerical columns
cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
num_cols = [col for col in df.columns if col not in cat_cols + ["Year", "Selling_Price"]]

# Handle missing values
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype("category")

# Create Car_Age and drop Year
df["Car_Age"] = 2020 - df["Year"]
df = df.drop(columns=["Year"])

# Remove extreme outliers in Selling_Price
price_threshold = df["Selling_Price"].quantile(0.95)
df = df[df["Selling_Price"] < price_threshold]

# ✅ Apply log transformation to Selling_Price
df["Selling_Price"] = np.log1p(df["Selling_Price"])  # log(1 + price) to handle zeros safely

# Save cleaned file with versioning
df.to_csv("cleaned_car_data_v1.csv", index=False)
print("Data cleaned & log-transformed. Saved as 'cleaned_car_data_v1.csv'")
