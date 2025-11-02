import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

df = pd.read_csv('datasets/ny_hp_cleaned.csv')

numeric_cols = ['lot_size', 'waterfront', 'age','land_value', \
                'new_construct', 'central_air', 'living_area', \
                    'pct_college', 'bedrooms', 'fireplaces', \
                        'bathrooms', 'rooms', 'test']
categorical_cols = ['fuel_type', 'heat_type', 'sewer_type']

X = df.drop('price', axis=1)
y = df['price']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Use the default rbf kernel
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR())
])

#random states for robustness testing
random_states = [42, 67, 314, 2025, 505, 2718, 777, 404, 911, 420]
RMSE_list = []
MAE_list = []
r2_list = []
best_params_list = []
param_grid = {
    'regressor__kernel': ['rbf', 'poly'],
    'regressor__C': [1, 10, 100],
    'regressor__gamma': ['scale', 0.1, 0.01],
    'regressor__epsilon': [0.1, 0.2, 0.5]
}

for current_random_state in random_states:
    print(f"\nRandom State: {current_random_state}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=current_random_state)
    
    grid_search = GridSearchCV(
        svm_pipeline, param_grid,
        cv=3, # 3-fold CV for speed; can increase to 5
        scoring='r2', # Optimize for R²
        n_jobs=-1, # Use all CPU cores
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_params_list.append(best_params)
    
    y_pred = best_model.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMS Error: {rmse:.2f} randomstate: {current_random_state}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.3f}")
    RMSE_list.append(rmse)
    MAE_list.append(mae)
    r2_list.append(r2)

print("\n================ Summary Across Random States ================")
print(f"Avg RMSE: {np.mean(RMSE_list):.2f}")
print(f"Avg MAE: {np.mean(MAE_list):.2f}")
print(f"Avg R²: {np.mean(r2_list):.3f}")
print("\nBest parameters per random state:")
for rs, params in zip(random_states, best_params_list):
    print(f" RS {rs}: {params}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', color='dodgerblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2, label='Ideal fit (y = x)')
plt.xlabel('Actual House Price', fontsize=12)
plt.ylabel('Predicted House Price', fontsize=12)
plt.title('SVR Predictions vs Actual Prices', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the figure
plt.savefig('plots/svr_predictions_vs_actual.png', dpi=300)
plt.show()
print("✅ Saved plot as 'svr_predictions_vs_actual.png'")