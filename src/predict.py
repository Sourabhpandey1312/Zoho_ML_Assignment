def predict(model, new_data):
    """
    Predict the audience rating for new data.

    Args:
        model (Pipeline): Trained model.
        new_data (pd.DataFrame): New data for prediction.

    Returns:
        float: Predicted audience rating.
    """
    return model.predict(new_data)