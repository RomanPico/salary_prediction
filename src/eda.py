import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


'''
Here we define some functions that perform EDA on the given datasets. This module includes two kind of functions:
- Functions that check the datasets for inconsistencies or null values. The idea is to verify the integrity of the given datasets.
- Functions that helps to visualize the datasets. The idea is to plot different variables and see how is their structure.

The idea of the first group of functions is to guide  us to create a preprocessing module to clean the datasets. In future versions the idea
is to enhance the preprocessing module so we can not just delete all the rows wilth null values, but replace them with different values (following logical rules, of course).

The second group of functions will help us to understand the data and how its different categories relate and how to preprocess the data. As a first step, it was found that
    the Salary column is not normally distributed, so we will use the log of the Salary column to make it more normally distributed. In future versions, the idea is to find
    correlations between other variables and see if we can use a better method that one shot encoding.

'''
def check_id_consistency(df_people, df_salary):
    """
    Here we just check that both datafiles have the same id.
    """
    ids_people = set(df_people["id"])
    ids_salary = set(df_salary["id"])

    only_in_people = ids_people - ids_salary
    only_in_salary = ids_salary - ids_people

    print(f"🔎 IDs in people.csv which are not in salary.csv: {len(only_in_people)}")
    print(f"🔎 IDs in salary.csv which are not in people.csv: {len(only_in_salary)}")

def check_nulls(df, name="DataFrame"):
    """
    Here we sum all the null values for every column to understand how incomplete is the data .
    """
    print(f"\n📋 amount of nulls per column {name}:")
    print(df.isnull().sum())

def count_salary_nulls(df):
    """
    Salary is the target variable, so it's really important. If a row has not a salary, we can't use it for training.
    """
    null_salary = df["Salary"].isnull().sum()
    print(f"\n rows in salary that are null: {null_salary}")

def count_rows_with_any_null(df, name="DataFrame"):
    """
    Since the dataset is "small", we can see directly which cells are NaN. This help us decide
    how to handle them better. If the dataset was bigger, this function maybe it's not so useful.
    """
    null_rows = df[df.isnull().any(axis=1)]
    print(f"\n amount of rows that have at least a NaN value: {len(null_rows)}")
    print(null_rows)


def print_df_overview(df, name="DataFrame"):
    """
    Print the dtype of the different columns. We can check if a "numerical variable" is not float type. 
    """
    print("\n dtype:")
    print(df.dtypes)

def plot_distributions(df):
    """ 
    As data is better understood with graphs, we plot the distributions of some variables.
    Also, as the Salary column is not normally distributed, we try the natural logarithm.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    sns.histplot(df["Salary"], bins=30, kde=True, ax=axes[0])
    axes[0].set_title("Salary")

    sns.histplot(df["Age"], bins=20, kde=True, ax=axes[1])
    axes[1].set_title("Age")

    sns.histplot(df["Years of Experience"], bins=20, kde=True, ax=axes[2])
    axes[2].set_title("Years of Experience")

    sns.histplot(df["Salary_log"], bins=30, kde=True, ax=axes[3])
    axes[3].set_title("ln(Salary)")

    plt.tight_layout()
    plt.show()

def count_job_titles(df, threshold):
    """
    The categorical variable job title has a lot of different values. We need to understand how this values
    are distributed. We use threshold so we can "change" how we group them.
    """
    job_counts = df["Job Title"].value_counts()
    titles_above_N = job_counts[job_counts > threshold]
    total_rows_above_N = titles_above_N.sum()

    print(f"\n Amount of rows with job titles that appear more than {threshold} times: {total_rows_above_N}")
    print(f" Job Titles with more than {threshold} repetitions:\n{titles_above_N}")