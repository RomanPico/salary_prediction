import pandas as pd
import numpy as np
from src.features import transform_features

def predict_from_csv(filepath, model, job_threshold=3):
    """
    Predicts salaries using the trained model from a new CSV input.

    The input file must contain the same features used during training:
    - 'Age', 'Education Level', 'Job Title', 'Years of Experience'

    Args:
        filepath (str): Path to CSV file with new inputs (no Salary column required).
        model (object): Trained regression model (must have .feature_names_in_ attribute).
        job_threshold (int): Threshold for grouping rare job titles (must match training).

    Returns:
        pd.DataFrame: Input data with an added column for predicted salaries.
    """
    # Load the new data
    df_new = pd.read_csv(filepath)
    
    ##Check for unseen education levels
    training_levels = {"Bachelor's", "Master's", "PhD"}  # hardcoded from the training data
    input_levels = set(df_new["Education Level"].dropna().unique())

    unexpected_levels = input_levels - training_levels
    if unexpected_levels:
        print(" WARNING: The following education levels were not used during training and will make the model perform poorly:")
        print("", ", ".join(unexpected_levels))

    #Apply same transformations as in training
    output = transform_features(df_new, job_threshold=job_threshold)

    #handle variable return (X, y) or just X
    if isinstance(output, tuple):
        X_new = output[0]
    else:
        X_new = output

    # Align columns with those used during training
    X_new = X_new.reindex(columns=model.feature_names_in_, fill_value=0)

    #predict log-salaries and revert the log transform
    y_pred_log = model.predict(X_new)
    y_pred_real = np.exp(y_pred_log)

    # Add predictions to the original DataFrame
    df_new = df_new.copy()
    df_new["Predicted Salary"] = y_pred_real

    return df_new

    