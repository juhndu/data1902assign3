import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

#Alternate version of gradient boosting that manually implements grid search
#with early stopping via post-facto evaluation
#Much slower than using built-in early stopping in GridSearchCV

#load and prep data
df = pd.read_csv('datasets/ny_hp_cleaned.csv')
print(df.head())

categorical_cols = ['fuel_type', 'heat_type', 'sewer_type']
for col in categorical_cols:
    df[col] = df[col].astype('category')

X = df.drop('price', axis=1)
y = df['price']

#manual param search to implement "early stopping"
param_grid = {
    'n_estimators': [600],  # max cap; actual "best round" chosen post-facto
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [1, 3, 5]
}


random_states = [42, 67, 314, 2025, 505, 2718, 777, 404, 911, 420]
RMSE_list, MAE_list, R2_list = [], [], []

#function for post-facto early stopping evaluation
def evaluate_params(params, X_train, y_train, random_state):
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    val_rmse_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            enable_categorical=True,
            random_state=random_state,
            eval_metric='rmse',
            **params
        )

        #train with full num of estimators and record all eval results
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        #extract RMSE at each boosting round
        evals_result = model.evals_result()
        val_errors = evals_result['validation_0']['rmse']
        best_rmse = min(val_errors)   # post-facto early stopping
        val_rmse_scores.append(best_rmse)

    return np.mean(val_rmse_scores)


#main loop over random states
for current_random_state in random_states:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=current_random_state
    )

    best_rmse = float("inf")
    best_params = None

    #Manual grid search with post-facto early stopping
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            for subsample in param_grid['subsample']:
                for colsample_bytree in param_grid['colsample_bytree']:
                    for reg_lambda in param_grid['reg_lambda']:
                        for n_estimators in param_grid['n_estimators']:

                            params = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'learning_rate': learning_rate,
                                'subsample': subsample,
                                'colsample_bytree': colsample_bytree,
                                'reg_lambda': reg_lambda
                            }

                            mean_rmse = evaluate_params(params, X_train, y_train, current_random_state)
                            print(f"Params: {params}, Mean (post-facto) CV RMSE: {mean_rmse:.4f}")

                            if mean_rmse < best_rmse:
                                best_rmse = mean_rmse
                                best_params = params

    print(f"\nBest Parameters for random_state={current_random_state}:\n{best_params}")
    print(f"Best post-facto CV RMSE: {best_rmse:.4f}")

    #Train final model using early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=current_random_state
    )

    final_model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        enable_categorical=True,
        random_state=current_random_state,
        eval_metric='rmse',
        early_stopping_rounds=20,
        **best_params
    )

    final_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred = final_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nFinalModel RMSE on test set: {rmse:.4f} (random_state={current_random_state})")

    RMSE_list.append(rmse)
    MAE_list.append(mae)
    R2_list.append(r2)


print("\n=======================")
print(f"Average RMSE across seeds: {np.mean(RMSE_list):.4f}")
print(f"Average MAE across seeds:  {np.mean(MAE_list):.4f}")
print(f"Average R² across seeds:   {np.mean(R2_list):.4f}")
print("=======================")