import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import category_encoders as ce

# Load data
df = pd.read_csv("cleaned_car_data.csv")

# Features and target
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

# Define categorical and numerical columns
cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
num_cols = [col for col in X.columns if col not in cat_cols]

# Target Encoder
encoder = ce.TargetEncoder(cols=cat_cols)

# Simpler, more generalizable model
model = RandomForestRegressor(
    random_state=42,
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=4
)

pipe = Pipeline([
    ('encoder', encoder),
    ('model', model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Metrics
print("\nFirst 10 Predictions:\n")
print(pd.DataFrame({'Actual': y_test.values[:10], 'Predicted': y_pred[:10]}).round(2))

train_r2 = pipe.score(X_train, y_train)
test_r2 = pipe.score(X_test, y_test)

print(f"\nTrain R²: {train_r2:.2f}")
print(f"Test R²: {test_r2:.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

X_test_with_preds = X_test.copy()
X_test_with_preds["Actual"] = y_test.values
X_test_with_preds["Predicted"] = y_pred
grouped = X_test_with_preds.groupby("Car_Age")[["Actual", "Predicted"]].mean().round(2)

grouped.plot(kind='bar', figsize=(10, 5), colormap='Paired')
plt.title("Average Actual vs Predicted Price by Car Age")
plt.xlabel("Car Age (Years)")
plt.ylabel("Average Selling Price (in Lakhs)")
plt.tight_layout()
plt.show()
