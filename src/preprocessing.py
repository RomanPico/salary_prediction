import pandas as pd
import numpy as np

def prepare_data(people_path, salary_path):
    """
    Load and clean the data
    
    - reads the CSV files for people and salary.
    - merges them based on ID.
    - drops rows with null values.
    - adds a new column: log(Salary)
    
    Returns:
    - df_clean: DataFrame with the cleaned data and log(Salary) column.
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
