# imports for building a knn model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_split import split_data
from backward_selection import backward_selected_knn
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


#split feature engineered data into train val test
data = pd.read_csv('datasets/ny_hp_feature_engineered.csv')
#split_data(dataset, y, test_size, val_size, random_state)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, 'price', 0.2, 0.2, 42)

#drop log price
X_train = X_train.drop(columns=['price_log'])
X_val = X_val.drop(columns=['price_log'])
X_test = X_test.drop(columns=['price_log'])

#standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#rejoin scaled data with y
train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train = pd.concat([train_scaled, y_train.reset_index(drop=True)], axis=1)  
val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
val = pd.concat([val_scaled, y_val.reset_index(drop=True)], axis=1)
test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test = pd.concat([test_scaled, y_test.reset_index(drop=True)], axis=1)

#log transform price
train['price_log'] = np.log1p(train['price'])
log_y_train = train['price_log']
val['price_log'] = np.log1p(val['price'])
log_y_val = val['price_log']
test['price_log'] = np.log1p(test['price'])
log_y_test = test['price_log']


#interaction terms
X_train['interaction_1'] = X_train['living_area'] * X_train['rooms']
X_train['interaction_2'] = X_train['living_area'] * X_train['land_value']
X_train['interaction_3'] = X_train['waterfront'] * X_train['land_value']
X_train['interaction_4'] = X_train['living_area_log'] * X_train['rooms']
X_train['interaction_5'] = X_train['living_area_log'] * X_train['land_value_log']
X_train['interaction_6'] = X_train['waterfront'] * X_train['land_value_log']

X_val['interaction_1'] = X_val['living_area'] * X_val['rooms']
X_val['interaction_2'] = X_val['living_area'] * X_val['land_value']
X_val['interaction_3'] = X_val['waterfront'] * X_val['land_value']
X_val['interaction_4'] = X_val['living_area_log'] * X_val['rooms']
X_val['interaction_5'] = X_val['living_area_log'] * X_val['land_value_log']
X_val['interaction_6'] = X_val['waterfront'] * X_val['land_value_log']

X_test['interaction_1'] = X_test['living_area'] * X_test['rooms']
X_test['interaction_2'] = X_test['living_area'] * X_test['land_value']
X_test['interaction_3'] = X_test['waterfront'] * X_test['land_value']  
X_test['interaction_4'] = X_test['living_area_log'] * X_test['rooms']
X_test['interaction_5'] = X_test['living_area_log'] * X_test['land_value_log']
X_test['interaction_6'] = X_test['waterfront'] * X_test['land_value_log']    


predictors = ['bathrooms', 'waterfront', 'fuel_type_Oil', 'new_construct', 'living_area_log', 'land_value_log']

neighbours=np.arange(1, 101)
best_score = -np.inf
    
for k in neighbours: 
    knn = KNeighborsRegressor(n_neighbors = k, metric='euclidean') 
    scores = cross_val_score(knn, X_train[predictors], y_train, 
                             cv=10, scoring = 'neg_mean_squared_error')
    # taking the average of scores across 20 folds
    cv_score = np.mean(scores)
     # use the cv score for model selection
    if cv_score >= best_score:
        best_score = cv_score
        best_knn = knn
    
knn = best_knn
knn.fit(X_train[predictors], y_train)


predictions = knn.predict(X_val[predictors])
val_rmse = np.sqrt(mean_squared_error(y_val, predictions))
cv_rmse= np.sqrt(-best_score)
k =knn.n_neighbors

knn = KNeighborsRegressor(n_neighbors= k, metric='euclidean')
knn.fit(X_test[predictors], y_test)
predictions = knn.predict(X_test[predictors])

final_rmse = np.sqrt(mean_squared_error(y_test, predictions))
final_rsquared = r2_score(y_test, predictions)
final_mae = mean_absolute_error(y_test, predictions)

print(f" Final validation RMSE: {final_rmse:.4f}")
print(f" R²: {final_rsquared:.4f}")
print(f" MAE: {final_mae:.4f}")
