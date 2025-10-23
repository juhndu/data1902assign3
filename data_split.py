
from sklearn.model_selection import train_test_split

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


