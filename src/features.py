import pandas as pd

def transform_features(df, job_threshold=3):
    """
    V0.1: 
    Transform the "clean dataset" into a dataset ready for the model.
    Until now, the features we included are:
    -One hot encoding for categorical variable "Education Level".
    -Group Job titles with few appearances as "Other" following what we learned in EDA
    -One hot encoding for Job Title.
    """
    df_feat = df.copy()

    # group job titles with appareances lower than threshold in other.
    job_counts = df_feat["Job Title"].value_counts()
    titles_above_threshold = job_counts[job_counts > job_threshold].index
    df_feat["Job Title"] = df_feat["Job Title"].apply(
        lambda x: x if x in titles_above_threshold else "Other"
    )

    # One hot encoding using get.dummies
    df_encoded = pd.get_dummies(df_feat, columns=["Education Level", "Job Title"], drop_first=True)

    # we select and return the features and the target variable. 
    X = df_encoded.drop(columns=["id", "Salary", "Salary_log", "Gender", "description"], errors="ignore")
    y = df_encoded["Salary_log"]

    return X, y
