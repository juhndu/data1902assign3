import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
data = pd.read_csv('datasets/ny_hp_feature_engineered.csv')

# Feature engineering (if not already saved in file)
# Add interaction terms
for prefix in ['train', 'val', 'test']:  # Safe guard if not precomputed
    pass

# Predictors (as in your current workflow)
predictors = ['bathrooms', 'waterfront', 'fuel_type_Oil', 'new_construct',
              'living_area_log', 'land_value_log']

# Random states for repeated evaluation
random_states = [42, 67, 314, 2025, 505, 2718, 777, 404, 911, 420]

RMSE_list, MAE_list, R2_list = [], [], []

for rs in random_states:
    print(f"\n🔹 Random State: {rs}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data[predictors], data['price'], test_size=0.2, random_state=rs
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Find best K using cross-validation
    best_score = -np.inf
    best_k = None
    neighbours = np.arange(1, 101)

    for k in neighbours:
        knn = KNeighborsRegressor(n_neighbors=k, metric='euclidean')
        scores = cross_val_score(knn, X_train_scaled, y_train,
                                 cv=10, scoring='neg_mean_squared_error')
        cv_score = np.mean(scores)
        if cv_score >= best_score:
            best_score = cv_score
            best_k = k

    print(f"Best k found: {best_k}")

    # Train final KNN on full training data
    best_knn = KNeighborsRegressor(n_neighbors=best_k, metric='euclidean')
    best_knn.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = best_knn.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.3f}")

    RMSE_list.append(rmse)
    MAE_list.append(mae)
    R2_list.append(r2)

# Summary across all runs
print("\n================ Summary Across Random States ================")
print(f"Average RMSE: {np.mean(RMSE_list):.2f}")
print(f"Average MAE: {np.mean(MAE_list):.2f}")
print(f"Average R²: {np.mean(R2_list):.3f}")