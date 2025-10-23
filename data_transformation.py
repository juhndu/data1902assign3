# this page will help us create a final data set for train/val/test after initial cleaning and feature engineering

import pandas as pd

sydney_house_prices = pd.read_csv('datasets/cleaned_housedata_sydney_enriched.csv')



print(sydney_house_prices.columns)