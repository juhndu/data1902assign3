import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor


df = pd.read_csv('datasets/ny_hp_cleaned.csv')
print(df.head)

categorical_cols = ['fuel_type', 'heat_type', 'sewer_type']
for col in categorical_cols:
    df[col] = df[col].astype('category')

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb = XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    enable_categorical=True,  # no categorical cols in this dataset
    random_state=42
)

param_grid = {
    'n_estimators': [200, 400, 600],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [1, 3, 5]
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',  # sklearn uses negative loss for maximization
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best Parameters:\n", grid_search.best_params_)


best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"\nBest RMSE on test set: {rmse:.4f}")

residuals = y_test - y_pred
plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

sm.qqplot(residuals, line='45', fit=True)
plt.title('QQ Plot of Residuals')
plt.show()