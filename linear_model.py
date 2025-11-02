import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from data_split import split_data
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

# load data
data = pd.read_csv('datasets/ny_hp_cleaned.csv')
X = data.drop('price', axis=1)
y = data['price']

#10 random states for splitting the data into train and test
#Take avg of these 10 results for final metrics
random_states = [42, 67, 314, 2025, 505, 2718, 777, 404, 911, 420]
RMSE_list = []
MAE_list = []
r2_list = []

for current_random_state in random_states:
    # split data into train, test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=current_random_state
    )

    selected_variables = X_train.columns.tolist()
    model_formula = 'price ~ ' + ' + '.join(selected_variables)
    model_bm = smf.ols(formula=model_formula, data=pd.concat([X_train, y_train], axis=1)).fit()

    y_pred = model_bm.predict(X_test)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    #print(f'Baseline Model RMSE: {rmse} randomstate: {current_random_state}')
    RMSE_list.append(rmse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    #print(f'Baseline Model MAE: {mae} randomstate: {current_random_state}')
    MAE_list.append(mae)
    r2 = metrics.r2_score(y_test, y_pred)
    r2_list.append(r2)

    model_bm.summary()

print(f'Average Model RMSE: {np.mean(RMSE_list)}')
print(f'Average Model MAE: {np.mean(MAE_list)}')
print(f'Avg Model R² Score: {np.mean(r2_list)}')

print(y_test.shape)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', color='dodgerblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2, label='Ideal fit (y = x)')
plt.xlabel('Actual House Price', fontsize=12)
plt.ylabel('Predicted House Price', fontsize=12)
plt.title('MLR Predictions vs Actual Prices', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the figure
plt.savefig('plotmlr_predictions_vs_actual.png', dpi=300)
plt.show()
print("✅ Saved plot as 'mlr_predictions_vs_actual.png'")


#residual plot
y_pred = model_bm.predict(X_train)
residuals = y_train - y_pred
plt.figure(figsize=(10,6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values (Baseline Model)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig('plots/baseline_model/residuals_baseline_model.png')

### normality of residuals via qq plot
sm.qqplot(residuals, line ='s')
plt.title('QQ Plot of Residuals (Baseline Model)')
plt.savefig('plots/baseline_model/qqplot_residuals_baseline_model.png')