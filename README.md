# Salary Forecast V1.2

This proyect consist in a machine learning model for predicting the salary of a person. This is done using different variables like Age, education, job position and years of experience.

The main idea in the whole project is to build a simple but functional pipeline, flexible enought to include different features in future versions.

The project is structure in the following way:

- `main.ipynb`: Jupyter notebook where we load the data, train the model, and visualize the results.

- `src/`: Contains all the modular code used in the project:
  - `eda.py`: Performs exploratory data analysis on both the raw and cleaned datasets. This module is optional — the code runs fine without it. It has two parts: one for inspecting the original dataset and another for generating plots.
  - `preprocessing.py`: Data loading and cleaning
  - `features.py`: Transformation of categorical variables
  - `model.py`: Model training, evaluation, and performance metrics
  - `predict.py`: Allows user to calculate salary using the trained model
  - `optional_feats.py`: Includes saving model predictions in a SQL database.

- `data/`: Contains the original CSVs and an file for salary predictions

- `README.md`: This file
- `requirements.txt`: Libraries needed to run this code.


 _Intended Initial Features_

- Load and clean the dataset using simple heuristics (drop rows with missing values)
- Perform optional exploratory data analysis (EDA) including null inspection and variable distributions
- Transform features: one-hot encoding for categorical variables and log-transform for the target (Salary)
- Train a baseline predictive model using linear regression (Scikit-learn)
- Evaluate the model using MAE and RMSE
- Report 95% confidence intervals for the metrics using bootstrap resampling
- Compare model performance against a DummyRegressor (mean prediction)

_Aditional Features included post V1.0_

- Save the model predictions in a SQL database.
- Aditional graphs to visualize the relationship between used features and target variables.


 _Future Improvements_

- Add support for more sophisticated models (e.g., Random Forest)
- Improve missing data handling instead of dropping rows (e.g., imputation based on correlated variables)


##  How to run the project

Clone this repository:


git clone https://github.com/RomanPico/salary_prediction/.git
cd salary-forecast
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook main.ipynb


## CSV Format for Predictions

To generate predictions on new data, you must provide a CSV file (e.g., `predict_sample.csv`) located in the `data/` folder.

The file must include **only the following columns**, with **exact same column names** as during training:

- `Age` (numeric)
- `Education Level` (one of: `"Bachelor's"`, `"Master's"`, `"PhD"`, etc.)
- `Job Title` (string)
- `Years of Experience` (numeric)

Example:

Age Education Level Job Title Years of Experience

35,Bachelor's,Data Scientist,5

40,Master's,Senior Data Analyst,10

28,PhD,Research Scientist,3

##  Ethical Considerations

During feature selection, we explored whether including `Gender` could improve model accuracy. While using gender as a feature led to a small improvement (~1.5% reduction in MAE), we ultimately decided to exclude it from the model due to ethical concerns.

An exploratory analysis showed a noticeable difference in salary distributions between genders. However, we did not find strong evidence that the salary for a given role varies significantly by gender. A possible explanation is that men and women in the dataset are not evenly distributed across job positions, which may partially account for the observed difference.

Given the risk of reinforcing bias — especially if gender is correlated with access to higher-paying roles — we opted to keep the model gender-neutral. The analysis and plots are left commented in the notebook for transparency and reproducibility.

