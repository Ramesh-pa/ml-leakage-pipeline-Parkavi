# ml-leakage-pipeline-Parkavi
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Set seed for reproducibility
np.random.seed(42)

# Create synthetic dataset (50+ records)
n = 60
area_sqft = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 5, n)
age_years = np.random.randint(0, 30, n)

# Generate price (in lakhs) with some noise
price_lakhs = (
    area_sqft * 0.05 +
    num_bedrooms * 5 -
    age_years * 0.3 +
    np.random.normal(0, 5, n)
)

# Create DataFrame
df = pd.DataFrame({
    'area_sqft': area_sqft,
    'num_bedrooms': num_bedrooms,
    'age_years': age_years,
    'price_lakhs': price_lakhs
})

# Features & target
X = df[['area_sqft', 'num_bedrooms', 'age_years']]
y = df['price_lakhs']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Print intercept & coefficients
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Display first 5 actual vs predicted
result = pd.DataFrame({
    'Actual': y[:5],
    'Predicted': y_pred[:5]
})
print("\nFirst 5 Actual vs Predicted:")
print(result)    
