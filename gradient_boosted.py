import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor, plot_tree




df = pd.read_csv('datasets/ny_hp_cleaned.csv')
print(df.head)

categorical_cols = ['fuel_type', 'heat_type', 'sewer_type']
for col in categorical_cols:
    df[col] = df[col].astype('category')

X = df.drop('price', axis=1)
y = df['price']


random_states = [42, 67, 314, 2025, 505, 2718, 777, 404, 911, 420]
RMSE_list = []
MAE_list = []
r2_list = []

for current_random_state in random_states:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=current_random_state
    )

    xgb = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        enable_categorical=True,
        random_state=current_random_state
    )

    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_lambda': [1, 3, 5]
    }

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error', 
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    print("Best Parameters:\n", grid_search.best_params_)


    best_model = grid_search.best_estimator_

    #new model trained on all data + early stopping on validation set
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
        **grid_search.best_params_
    )
    final_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    y_pred = final_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"\nFinalModel RMSE on test set: {rmse:.4f} randomstate: {current_random_state}")
    RMSE_list.append(rmse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nFinalModel RMSE on test set: {rmse:.4f} randomstate: {current_random_state}")
    MAE_list.append(mae)
    r2 = r2_score(y_test, y_pred)
    r2_list.append(r2)

    #y_pred = best_model.predict(X_test)
    #rmse = root_mean_squared_error(y_test, y_pred)
    #print(f"\nBestModel RMSE on test set: {rmse:.4f}")

print(f'Average Model RMSE: {np.mean(RMSE_list)}')
print(f'Average Model MAE: {np.mean(MAE_list)}')
print(f'Average Model R2 Score: {np.mean(r2_list)}')

importances = final_model.get_booster().get_score(importance_type='gain')
sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print("Top important features:\n", sorted_feats[:10])

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', color='dodgerblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2, label='Ideal fit (y = x)')
plt.xlabel('Actual House Price', fontsize=12)
plt.ylabel('Predicted House Price', fontsize=12)
plt.title('XGB Predictions vs Actual Prices', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the figure
plt.savefig('plots/xgb_predictions_vs_actual.png', dpi=300)
plt.show()
print("✅ Saved plot as 'xgb_predictions_vs_actual.png'")


#Choose which trees to visualize
num_trees = final_model.get_booster().num_boosted_rounds()
print(f"Total trees in model: {num_trees}")

# You can visualize, for example, the first few or trees where top features appear most
trees_to_plot = [0, 1, 2, 3, 4]  # or pick based on your analysis

#Plot trees inline using matplotlib
plt.figure(figsize=(20, 10))
plot_tree(final_model, num_trees=trees_to_plot[0], rankdir='LR')
plt.title(f"Tree {trees_to_plot[0]}")
plt.savefig(f'plots/trees/treeinline_{trees_to_plot[0]}.png')

# Optionally, plot several trees
for tree_idx in trees_to_plot[1:3]:
    plt.figure(figsize=(20, 10))
    plot_tree(final_model, num_trees=tree_idx, rankdir='LR')
    plt.title(f"Tree {tree_idx}")
    plt.savefig(f'plots/trees/tree_{tree_idx}.png')


#residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10,6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values (Baseline Model)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig('plots/baseline_model/residuals_xgb_model.png')

### normality of residuals via qq plot
sm.qqplot(residuals, line ='s')
plt.title('QQ Plot of Residuals (Baseline Model)')
plt.savefig('plots/baseline_model/qqplot_residuals_xgb_model.png')
