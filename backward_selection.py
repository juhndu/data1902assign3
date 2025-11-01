# imports for backward selection using KNN
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import statsmodels.api as sm


def backward_selected_knn(X_train, X_val, y_train, y_val, predictors, k):
    knn = KNeighborsRegressor(n_neighbors=k, metric='euclidean')
    knn.fit(X_train[predictors], y_train)
    predictions = knn.predict(X_val[predictors])
    current_rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f"Initial RMSE with all listed predictors: {current_rmse:.4f}")

    remaining = predictors.copy()

    while len(remaining) > 1:
        scores_with_candidates = []

        for feature in remaining:
            trial_features = [f for f in remaining if f != feature]

            X_train_sub = X_train[trial_features]
            X_val_sub = X_val[trial_features]

            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X_train_sub, y_train)
            preds = knn.predict(X_val_sub)
            rmse = np.sqrt(mean_squared_error(y_val, preds))

            scores_with_candidates.append((rmse, feature))

        scores_with_candidates.sort()
        best_new_rmse, worst_feature = scores_with_candidates[0]

        if best_new_rmse < current_rmse:
            print(f"Removing {worst_feature} improved RMSE: {current_rmse:.4f} → {best_new_rmse:.4f}")
            remaining.remove(worst_feature)
            current_rmse = best_new_rmse
        else:
            break

    final_features = list(remaining)
    knn.fit(X_train[final_features], y_train)
    final_preds = knn.predict(X_val[final_features])

    final_rmse = np.sqrt(mean_squared_error(y_val, final_preds))
    final_rsquared = r2_score(y_val, final_preds)
    final_mae = mean_absolute_error(y_val, final_preds)

    print(f"\n Final features: {final_features}")
    print(f" Final validation RMSE: {final_rmse:.4f}")
    print(f" R²: {final_rsquared:.4f}")
    print(f" MAE: {final_mae:.4f}")

    return final_features


