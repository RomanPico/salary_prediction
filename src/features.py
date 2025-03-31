import pandas as pd

def transform_features(df, job_threshold=3):
    """
    Transforms the cleaned dataset into features and target for modeling.

    This function performs feature engineering steps such as:
    - Groups job titles with fewer than `job_threshold` occurrences into "Other".
    - Applies one-hot encoding to 'Education Level' and 'Job Title'.
    - Drops irrelevant columns like 'id', 'Gender', 'description', and original salary columns.

    Args:
        df (pd.DataFrame): Cleaned input DataFrame with required columns.
        job_threshold (int): Minimum number of occurrences to keep a job title separate.

    Returns:
        X (pd.DataFrame): Feature matrix ready for training.
        y (pd.Series): Target variable (log of salary).
    """
   
    
    
    df_feat = df.copy()

    job_counts = df_feat["Job Title"].value_counts()
    titles_above_threshold = job_counts[job_counts > job_threshold].index
    df_feat["Job Title"] = df_feat["Job Title"].apply(
        lambda x: x if x in titles_above_threshold else "Other"
    )

    df_encoded = pd.get_dummies(df_feat, columns=["Education Level", "Job Title"], drop_first=True)

    X = df_encoded.drop(columns=["id", "Salary", "Salary_log", "Gender", "description"], errors="ignore")
    #comment the two previous lines and uncomment the next two lines to include Gender in the training.
    
    #df_encoded = pd.get_dummies(df_feat, columns=["Education Level", "Job Title", "Gender"], drop_first=True)
    #X = df_encoded.drop(columns=["id", "Salary", "Salary_log", "description"], errors="ignore")


    if "Salary_log" in df_encoded.columns:
        y = df_encoded["Salary_log"]
        return X, y
    else:
        return X
    
