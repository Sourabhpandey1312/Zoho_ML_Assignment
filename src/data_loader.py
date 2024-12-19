import pandas as pd
import os

def load_data(file_path):

    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def clean_data(data):

    # Drop irrelevant columns if they exist
    if 'critics_consensus' in data.columns:
        data = data.drop(columns=['critics_consensus'])

    # Impute missing values
    data['runtime_in_minutes'] = data['runtime_in_minutes'].fillna(data['runtime_in_minutes'].median())
    data['genre'] = data['genre'].fillna('Unknown')
    data['directors'] = data['directors'].fillna('Unknown')
    data['writers'] = data['writers'].fillna('Unknown')
    data['cast'] = data['cast'].fillna('Unknown')
    data['studio_name'] = data['studio_name'].fillna('Independent')

    # Drop rows with missing target variable
    data = data.dropna(subset=['audience_rating'])

    # Remove duplicates
    data = data.drop_duplicates()

    # Save cleaned dataset
    output_dir = './data/processed'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    cleaned_data_path = f'{output_dir}/cleaned_data.csv'
    data.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved successfully at {cleaned_data_path}")

    return data