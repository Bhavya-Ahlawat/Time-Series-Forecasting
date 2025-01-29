import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

def preprocess_data(data_path):
    """Loads and preprocesses the Jena climate dataset."""
    df = pd.read_csv(data_path)
    df['Date Time'] = pd.to_datetime(df['Date Time'], format="%d.%m.%Y %H:%M:%S")
    df.set_index('Date Time', inplace=True)
    return df

def evaluate_model(y_true, y_pred):
    """Evaluates the model using RMSE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse