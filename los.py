import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the cleaned data
df_clean = pd.read_csv('los_data.csv')

# Define features (X) and target variable (y)
X = df_clean.drop(columns=['LOS'])
y = df_clean['LOS']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Define the Random Forest model
rf_model = RandomForestRegressor(random_state=0)

# Define hyperparameter grid for GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the Random Forest model with the best parameters
best_rf_model = RandomForestRegressor(random_state=42, **best_params)
best_rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Calculate the percentage of accuracy (normalized R-squared)
percentage_accuracy = r2 * 100

# Display evaluation metrics
print(f"Best Hyperparameters: {best_params}")
print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse}")
print(f"R-squared (R2) Score on Test Set: {r2}")
print(f"Mean Absolute Error (MAE) on Test Set: {mae}")
print(f"Percentage of Accuracy: {percentage_accuracy:.2f}%")
