from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def preprocess(self):
        """Preprocesses the data
        
        Returns:
        preprocessed data"""
        # Add preprocessing steps here
        return self.data
    
    
    def split_data(self, train_size=0.7, val_size=0.15):
        """Applies a train, test, validation split on the data and labels.
        
        Parameters:
        X           (list)  - The input data
        y           (list)  - The input labels
        train_size  (float) - Percentage of total data used for training
        val_size    (float) - Percentage of total data used for validation
        
        Returns:
        training data + labels
        validation data + labels
        testing data + labels"""

        X = self.data 
        y = self.labels
        test_size = 1 - train_size - val_size
        # Step 1: Split the data into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size)

        # Step 2: Split the train+val set into training and validation sets
        val_ratio = val_size / (train_size + val_size)  # Adjust val_size proportionally
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio)

        self.training_data = X_train
        self.training_labels = y_train
        self.validation_data = X_val
        self.validation_labels = y_val
        self.testing_data = X_test
        self.testing_labels = y_test
        return X_train, X_val, X_test, y_train, y_val, y_test
