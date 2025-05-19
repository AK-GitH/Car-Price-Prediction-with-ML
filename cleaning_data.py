import pandas as pd

df = pd.read_csv("car data.csv")
df = df.drop(columns=["Car_Name"], errors='ignore') # dropping car name column, as its useless
df = df.drop_duplicates() # removing dulpicate rows

cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
num_cols = [col for col in df.columns if col not in cat_cols + ["Year", "Selling_Price"]]

# handle missing values
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# make Car_Age and drop the original Year
df["Car_Age"] = 2020 - df["Year"]
df = df.drop(columns=["Year"])

# remove extreme outliers in Selling_Price 
price_threshold = df["Selling_Price"].quantile(0.95)
df = df[df["Selling_Price"] < price_threshold]


df.to_csv("cleaned_car_data.csv", index=False)
print("Data has been cleaned & saved as 'cleaned_car_data.csv'")