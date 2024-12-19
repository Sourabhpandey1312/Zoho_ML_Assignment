import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import IterativeImputer

def get_preprocessor(categorical_cols, numerical_cols, advanced_impute=False):
    """
    Create a preprocessing pipeline.

    ColumnTransformer: Preprocessing pipeline.
    """
    if advanced_impute:
        num_transformer = Pipeline(steps=[
            ('imputer', IterativeImputer(max_iter=10, random_state=42)),
            ('scaler', StandardScaler())
        ])
    else:
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer(transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])