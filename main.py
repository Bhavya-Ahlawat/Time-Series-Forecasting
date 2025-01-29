import matplotlib.pyplot as plt
import pandas as pd
from utils import preprocess_data, evaluate_model
from models import train_arima_model
import kagglehub
import warnings
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

# Download the dataset if it doesn't exist
try:
    with open('data/jena_climate_2009_2016.csv', 'r') as f:
        pass
except FileNotFoundError:
    print("Downloading dataset...")
    path = kagglehub.dataset_download("mnassrib/jena-climate")
    data_path = path + '/jena_climate_2009_2016.csv'
else:
    data_path = 'data/jena_climate_2009_2016.csv'

# Preprocess the data
df = preprocess_data(data_path)

# Select the target variable
target_variable = 'T (degC)'

# Split the data
train_data = df[target_variable][:int(0.8 * len(df))]
test_data = df[target_variable][int(0.8 * len(df)):]

# Train the ARIMA model
arima_model = train_arima_model(train_data)

# Make predictions
arima_predictions = arima_model.forecast(steps=len(test_data))

# Evaluate the model
arima_rmse = evaluate_model(test_data, arima_predictions)
print(f'ARIMA RMSE: {arima_rmse}')


plt.figure(figsize=(12, 6))

# Plot actual values for the entire range
plt.plot(df.index, df[target_variable], label='Actual', zorder=1) #zorder to put actual values on top

# Plot predictions only for the test data range
plt.plot(test_data.index, arima_predictions, label='ARIMA Predicted', zorder=2) #zorder to put predictions on top

plt.legend()
plt.title('Time Series Forecasting with ARIMA')
plt.xlabel('Time')
plt.ylabel(target_variable)

# Format x-axis to show dates nicely (Important for Time Series)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30)) # Show ticks every 30 days
plt.gcf().autofmt_xdate() # Rotate date labels for readability

plt.show()

# Example of printing model summary
print(arima_model.summary())