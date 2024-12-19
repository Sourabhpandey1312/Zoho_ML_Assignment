import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr

def eda(data):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.
    """
    # Check if 'critics_consensus' column exists and drop if necessary
    if 'critics_consensus' in data.columns:
        data.drop(columns=['critics_consensus'], inplace=True)

    # Distribution of the target variable
    plt.figure(figsize=(8, 6))
    sns.histplot(data['audience_rating'], kde=True, bins=20)
    plt.title("Distribution of Audience Ratings")
    plt.xlabel("Audience Rating")
    plt.ylabel("Frequency")
    plt.show()

    # Correlation heatmap for numerical features
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()