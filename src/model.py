from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor
import numpy as np

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        test_size (float): Fraction of data to use as test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, model=None):
    """
    Trains a regression model on the training data.

    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Target values for training.
        model (object, optional): Scikit-learn compatible model. Defaults to LinearRegression.

    Returns:
        object: Trained model.
    """

    if model is None:
        model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, show_examples=True, baseline=True):
    """
    Evaluates a trained model using MAE and RMSE (with confidence intervals).
    Also compares it against a DummyRegressor baseline.

    Args:
        model (object): Trained regression model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True target values (log-salaries).
        show_examples (bool): Whether to print sample predictions (currently disabled).
        baseline (bool): Whether to compare against a DummyRegressor.

    Returns:
        None
    """

    y_pred_log = model.predict(X_test)
    y_pred_real = np.exp(y_pred_log)
    y_test_real = np.exp(y_test)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    # MAE and RMSE using bootstrap
    mae_ci = bootstrap(y_test_real.values, y_pred_real, mean_absolute_error)
    rmse_ci = bootstrap(y_test_real.values, y_pred_real, lambda y1, y2: np.sqrt(mean_squared_error(y1, y2)))
   
    print("\n performance on test:")
    print("----------------------------------")
    print(" Model Evaluation Summary:")
    print("----------------------------------")
    print(f" MAE: ${mae:,.2f} (on average, predictions deviate this much from actual salaries)")
    print(f"95% Confidence Interval for MAE: ${mae_ci[0]:,.2f} – ${mae_ci[1]:,.2f}\n")

    print(f"RMSE: ${rmse:,.2f} ")
    print(f"95% Confidence Interval for RMSE: ${rmse_ci[0]:,.2f} – ${rmse_ci[1]:,.2f}\n")



    '''
    #ONLY for testing purposes
    if show_examples:
        print("\n Comparation of real salaries (test vs prediction):")
        for real, pred in zip(y_test_real[:5], y_pred_real[:5]):
            print(f"Real: ${real:,.2f}  |  Predicho: ${pred:,.2f}")
    '''
    if baseline:
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_test, y_test)
        dummy_preds = np.exp(dummy.predict(X_test))

        dummy_mae = mean_absolute_error(y_test_real, dummy_preds)
        dummy_rmse = np.sqrt(mean_squared_error(y_test_real, dummy_preds))

    improvement_mae = dummy_mae - mae
    improvement_rmse = dummy_rmse - rmse
    print("Comparison vs baseline (dum. regressor using mean):")
    print("--------------------------------------------------------")
    print(f" MAE (Dummy): ${dummy_mae:,.2f} -  improvement: ${improvement_mae:,.2f}")
    print(f" RMSE (Dummy): ${dummy_rmse:,.2f}  -  improvement: ${improvement_rmse:,.2f}")


        
def bootstrap(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95):
    """
    Computes a confidence interval for a metric using bootstrap resampling.

    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted values.
        metric_fn (function): Metric function to apply (e.g., mean_absolute_error).
        n_bootstrap (int): Number of bootstrap samples.
        ci (int): Confidence interval percentage (e.g., 95).

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """

    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(range(n), size=n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    alpha = (100 - ci) / 2
    lower = np.percentile(scores, alpha)
    upper = np.percentile(scores, 100 - alpha)

    return lower, upper

