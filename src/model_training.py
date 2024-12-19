from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
# from xgboost import XGBRegressor

def get_models(preprocessor):
    """
    Define a set of models to evaluate.

    Args:
        preprocessor (ColumnTransformer): Preprocessing pipeline.

    Returns:
        dict: Dictionary of model names and pipelines.
    """
    return{
        'RandomForest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))
        ]),
        'GradientBoosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42, n_estimators=100))
        ]),
        'LinearRegression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegrẻ̉̉ssion())
        ]),
        ''''XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(random_state=42))
        ]),
        'LightGBM': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LGBMRegressor(random_state=42))
        ]),'''
        'SVR': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SVR(kernel='rbf'))
        ])
    }

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple models.

    Args:
        models (dict): Dictionary of model pipelines.
        X_train, X_test (pd.DataFrame): Training and testing features.
        y_train, y_test (pd.Series): Training and testing targets.

    Returns:
        dict: Evaluation metrics for each model.
    """
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        print(f"Evaluated {name}: RMSE={rmse:.2f}, R2={r2:.2f}")

    # Save results to JSON
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results

def select_best_model(models, results):
    """
    Select the best model based on R² score.

    Args:
        models (dict): Dictionary of model pipelines.
        results (dict): Evaluation metrics for each model.

    Returns:
        str, Pipeline: Name and pipeline of the best model.
    """
    best_model_name = max(results, key=lambda name: results[name]['R2'])
    print(f"Best Model: {best_model_name}")
    return best_model_name, models[best_model_name]

def feature_importance(model, feature_names, model_name):
    """
    Display feature importance for tree-based models.

    Args:
        model (Pipeline): Trained model.
        feature_names (list): Feature names after preprocessing.
        model_name (str): Name of the model.

    Returns:
        None
    """
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        importances = model.named_steps['regressor'].feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[sorted_idx], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=90)
        plt.title(f"Feature Importance ({model_name})")
        plt.show()
    else:
        print(f"Feature importance is not available for {model_name}.")

