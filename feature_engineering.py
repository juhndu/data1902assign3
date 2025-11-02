## conduct feature engineering based of EDA findings

import pandas as pd
import numpy as np

data = pd.read_csv('datasets/ny_hp_cleaned.csv')

data['living_area_log'] = data['living_area'].apply(lambda x: np.log(x) if x > 0 else 0)
data['land_value_log'] = data['land_value'].apply(lambda x: np.log(x) if x > 0 else 0)  
data['price_log'] = data['price'].apply(lambda x: np.log(x) if x > 0 else 0)

#encoding categorical variables
data = pd.get_dummies(data, columns=['fuel_type', 'heat_type', 'sewer_type'], drop_first=True)

#print(data.head())

#save the engineered dataset
data.to_csv('datasets/ny_hp_feature_engineered.csv', index=False)



