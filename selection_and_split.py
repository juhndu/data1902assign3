# this page will help us create a final data set for train/val/test after initial cleaning and feature engineering

import pandas as pd
from data_split import split_data
from sklearn.model_selection import train_test_split

dataset1 = pd.read_csv('datasets/cleaned_housedata_sydney_enriched.csv') #<- this has the most predictors AND rows
dataset2 = pd.read_csv('datasets/nsw_propertydata_sydney_cleaned.csv')
dataset3 = pd.read_csv('datasets/suburb_yearly_bybednum_sydney_enriched.csv')

# print(f"Dataset1 shape: {dataset1.shape}")
# print(f"Dataset2 shape: {dataset2.shape}")
# print(f"Dataset3 shape: {dataset3.shape}")  

split_data(dataset1, 'price', test_size=0.2, val_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2    
