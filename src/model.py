from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor
import numpy as np

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Standard separation of the data in training and testing .
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, model=None):
    """
    Training the model. In future versions we look forward to use other models.
    """
    if model is None:
        model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, show_examples=True, baseline=True):
    """
    This function evaluates the model using the test set and calculates different metrcics. 
    We also compare with dummyregresor (using the mean) to see if our model is better
    than using the average.
    """
    y_pred_log = model.predict(X_test)
    y_pred_real = np.exp(y_pred_log)
    y_test_real = np.exp(y_test)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    # MAE and RMSE using bootstrap
    mae_ci = bootstrap(y_test_real.values, y_pred_real, mean_absolute_error)
    rmse_ci = bootstrap(y_test_real.values, y_pred_real, lambda y1, y2: np.sqrt(mean_squared_error(y1, y2)))
   
    #print(f" MAE : {mae:,.2f} $ |  95% CI: {mae_ci[0]:,.2f} – {mae_ci[1]:,.2f}")
    #print(f" RMSE: {rmse:,.2f} $ |  95% CI: {rmse_ci[0]:,.2f} – {rmse_ci[1]:,.2f}")
    print("\n performance on test:")
    print("----------------------------------")
    print(f" MAE: ${mae:,.2f}")
    print(f"Average prediction ${mae:,.0f}")
    print(f"95% confidence interval: ${mae_ci[0]:,.2f} – ${mae_ci[1]:,.2f}\n")

    print(f" RMSE : ${rmse:,.2f}")
    print(f"95% confidence interval: ${rmse_ci[0]:,.2f} – ${rmse_ci[1]:,.2f}\n")


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

        print(" baseline model (DummyRegressor – mean prediction):")
        print("-----------------------------------------------------")
        print(f" MAE (dummy): ${dummy_mae:,.2f}")
        print(f" RMSE (dummy): ${dummy_rmse:,.2f}")

        
def bootstrap(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95):
    """
    This function calculates the trust interval for a given metric using bootstrap method
    
    Parameters:
    -- y_true: real salaries values
    -- y_pred: predicted salaries values
    -- metric_fn: function that calculates the metric (eg: mean_absolute_error)
    -- n_bootstrap: number of random samples
    -- ci: confidence level (eg: 95)
    It returns the lower and upper limit of the confidence interval.
 
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

