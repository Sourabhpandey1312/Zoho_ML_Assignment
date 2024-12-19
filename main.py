import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data, clean_data
from src.eda import eda
from src.preprocessing import get_preprocessor
from src.model_training import get_models, evaluate_models, select_best_model
from src.predict import predict

# Ensure outputs directory exists
os.makedirs('./outputs', exist_ok=True)

# Load and Clean Data
data_path = './data/Rotten_Tomatoes_Movies3.csv'
data = load_data(data_path)
data = clean_data(data)

# Perform EDA
eda(data)

# Feature Selection
features = ['rating', 'genre', 'directors', 'runtime_in_minutes', 
            'studio_name', 'tomatometer_rating', 'tomatometer_count']
target = 'audience_rating'

X = data[features]
y = data[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing Pipeline
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
preprocessor = get_preprocessor(categorical_cols, numerical_cols)

# Define and Evaluate Models
models = get_models(preprocessor)
results = evaluate_models(models, X_train, X_test, y_train, y_test)

# Save Results to JSON
metrics_path = './outputs/metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Metrics saved to {metrics_path}")

# Select Best Model
best_model_name, best_model = select_best_model(models, results)

# Save Best Model
model_path = './outputs/best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"Best Model ({best_model_name}) saved to {model_path}")

# Predict on New Data
new_data = pd.DataFrame({
    'rating': ['PG-13'],
    'genre': ['Action'],
    'directors': ['Christopher Nolan'],
    'runtime_in_minutes': [150],
    'studio_name': ['Warner Bros.'],
    'tomatometer_rating': [92],
    'tomatometer_count': [400]
})
prediction = predict(best_model, new_data)
print(f"\nPredicted Audience Rating by {best_model_name}: {prediction[0]:.2f}")