
from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(dataset, y, test_size, val_size, random_state):

    # Extract target and features
    target_y = dataset[y]
    features_x = dataset.drop(columns=[y])

    # Step 1: split into train_full and test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features_x, target_y, test_size=test_size, random_state=random_state
    )

    # Step 2: split the train_full into train and validation

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test



dataset1 = pd.read_csv('datasets/cleaned_housedata_sydney_enriched.csv') #<- this has the most predictors AND rows
dataset2 = pd.read_csv('datasets/nsw_propertydata_sydney_cleaned.csv')
dataset3 = pd.read_csv('datasets/suburb_yearly_bybednum_sydney_enriched.csv')

# print(f"Dataset1 shape: {dataset1.shape}")
# print(f"Dataset2 shape: {dataset2.shape}")
# print(f"Dataset3 shape: {dataset3.shape}")  

X_train, X_val, X_test, y_train, y_val, y_test = split_data(dataset1, 'price', test_size=0.2, val_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2    

print(X_train.head())



