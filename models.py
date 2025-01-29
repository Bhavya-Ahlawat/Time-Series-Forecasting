from statsmodels.tsa.arima.model import ARIMA

def train_arima_model(train_data, order=(5, 1, 0)):
    """Trains an ARIMA model."""
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit