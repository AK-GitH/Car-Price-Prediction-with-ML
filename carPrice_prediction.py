import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("cleaned_car_data.csv")

X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
num_cols = [col for col in X.columns if col not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10)
pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# prediction results
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Car Prices (first 10 rows):\n")
print(results.head(10).round(2))

# metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# (bar) avg selling price by fuel type 
mean_prices_fuel = df.groupby("Fuel_Type")["Selling_Price"].mean().sort_values()

plt.figure(figsize=(8, 5))
sns.barplot(
    x=mean_prices_fuel.index,
    y=mean_prices_fuel.values,
    palette="Set2",
    edgecolor="black"
)
plt.title("Average Selling Price by Fuel Type", fontsize=14, fontweight="bold")
plt.xlabel("Fuel Type", fontsize=12)
plt.ylabel("Average Selling Price", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# (bar) avg selling price by transmission type 
mean_prices_trans = df.groupby("Transmission")["Selling_Price"].mean().sort_values()

plt.figure(figsize=(8, 5))
sns.barplot(
    x=mean_prices_trans.index,
    y=mean_prices_trans.values,
    palette="Paired",
    edgecolor="black"
)
plt.title("Average Selling Price by Transmission", fontsize=14, fontweight="bold")
plt.xlabel("Transmission", fontsize=12)
plt.ylabel("Average Selling Price", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()