import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class PreprocessingCleaning:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        self.data = data

    def check_missing_values(self):
        missing_values = self.data.isnull().sum()
        print("Missing Values:")
        print(missing_values)
        print("\nData Info:")
        print(self.data.info())
        print("\nData Description:")
        print(self.data.describe())

    def imputer_pipelines(self, numerical_features, categorical_features):
        # Define imputers
        numerical_imputer = SimpleImputer(strategy='mean')
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        # Create column transformer for pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_imputer, numerical_features),
                ('cat', categorical_imputer, categorical_features)
            ]
        )

        # Apply transformations
        self.data[numerical_features + categorical_features] = preprocessor.fit_transform(self.data)
        return self.data
