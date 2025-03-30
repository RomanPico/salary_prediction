import pandas as pd
import numpy as np

def prepare_data(people_path, salary_path):
    """
    Loads and cleans the dataset.

    This function merges the people and salary datasets using their 'id',
    removes any rows with missing values, and adds a new column with the
    logarithm of the salary.

    Args:
        people_path (str): Path to the people CSV file.
        salary_path (str): Path to the salary CSV file.

    Returns:
        pd.DataFrame: Cleaned dataset with an additional 'Salary_log' column.
    """

    
    # Cargar datasets
    df_people = pd.read_csv(people_path)
    df_salary = pd.read_csv(salary_path)

    # Merge por ID
    df_merged = df_people.merge(df_salary, on="id", how="left")

    # Eliminar filas con NaN
    df_clean = df_merged.dropna().copy()

    # Agregar columna con log del salario
    df_clean["Salary_log"] = np.log(df_clean["Salary"])

    return df_clean
