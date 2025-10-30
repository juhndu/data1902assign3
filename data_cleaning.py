import pandas as pd

data = pd.read_csv('datasets/ny_hp.csv')

# 1. check for nulls
print(data.info()) #<- nulls detected

#2.check for negative values in numercial columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    if (data[col] < 0).any():
        print(f"Negative values found in column: {col}") 
    else: 
        print(f"No negative values in column: {col}")

#no negative values found

# 3. drop rows with null values
data_cleaned = data.dropna()

#4. lower case column names
data_cleaned.columns = [col.lower() for col in data_cleaned.columns]
print(data_cleaned.columns)

#5. replace dots with underscores in column names
data_cleaned.columns = [col.replace('.', '_') for col in data_cleaned.columns]
print(data_cleaned.columns)

#save cleaned data
data_cleaned.to_csv('datasets/ny_hp_cleaned.csv', index=False)