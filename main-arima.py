# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error

# # Step 1: Load the data
# file_path = "combined_file_updated.csv"  # Update with your file path
# df = pd.read_csv(file_path)

# # Step 2: Preprocess the data
# # Convert 'Timestamp' to datetime format
# df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

# # Aggregate daily sales (Total sales amount per day)
# df['Date'] = df['Timestamp'].dt.date
# daily_sales = df.groupby('Date').agg(total_sales=('Amount', 'sum')).reset_index()

# # Step 3: Split the data into training and testing (rolling forecast will be applied to the test data)
# train_size = int(len(daily_sales) * 0.8)  # 80% train, 20% test
# train, test = daily_sales['total_sales'][:train_size], daily_sales['total_sales'][train_size:]

# # Step 4: Apply a rolling forecast with ARIMA
# # Set ARIMA parameters (p, d, q)
# p, d, q = 5, 1, 0  # Adjust based on ACF/PACF plots or trial and error

# history = [x for x in train]
# predictions = []
# test_dates = daily_sales['Date'][train_size:].reset_index(drop=True)

# for t in range(len(test)):
#     model = ARIMA(history, order=(p, d, q))
#     model_fit = model.fit()
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     history.append(test[t])

# # Step 5: Evaluate the model
# rmse = np.sqrt(mean_squared_error(test, predictions))
# print(f"Test RMSE: {rmse}")

# # Step 6: Visualize the predictions vs actual values
# plt.figure(figsize=(10, 6))
# plt.plot(test_dates, test, label='Actual')
# plt.plot(test_dates, predictions, label='Predicted', color='red')
# plt.title('Rolling ARIMA Forecast vs Actual')
# plt.xlabel('Date')
# plt.ylabel('Total Sales')
# plt.legend()
# plt.show()

# # Step 7: Predict future values (14/08/2024 - 28/08/2024) using rolling window
# # Start with the full dataset and continue rolling forecast into the future
# history = [x for x in daily_sales['total_sales']]  # Rebuild the full history
# future_predictions = []
# future_dates = pd.date_range(start="2024-08-14", end="2024-08-28", freq='D')

# for future_date in future_dates:
#     model = ARIMA(history, order=(p, d, q))
#     model_fit = model.fit()
#     output = model_fit.forecast()
#     yhat = output[0]
#     future_predictions.append(yhat)
#     history.append(yhat)  # Add the forecast to the history for the next iteration

# # Step 8: Visualize the future predictions
# plt.figure(figsize=(10, 6))
# plt.plot(future_dates, future_predictions, label='Future Predicted', color='green')
# plt.title('Future Sales Forecast (14/08/2024 - 28/08/2024)')
# plt.xlabel('Date')
# plt.ylabel('Predicted Total Sales')
# plt.legend()
# plt.show()

# # Step 9: Display the future forecasted values
# future_forecast = pd.DataFrame({
#     'Date': future_dates,
#     'Predicted_Sales': future_predictions
# })

# print("\nFuture Sales Forecast (14/08/2024 - 28/08/2024):")
# print(future_forecast)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Step 1: Load the data
file_path = "combined_file_updated.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Convert 'Timestamp' to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

# Aggregate daily sales (Total sales amount per day)
df['Date'] = df['Timestamp'].dt.date
daily_sales = df.groupby('Date').agg(total_sales=('Amount', 'sum')).reset_index()

# Step 3: Split the data into training and testing (rolling forecast will be applied to the test data)
train_size = int(len(daily_sales) * 0.8)  # 80% train, 20% test
train, test = daily_sales['total_sales'][:train_size], daily_sales['total_sales'][train_size:]

# Convert the test series to a list for easier indexing
test = test.tolist()

# Step 4: Apply a rolling forecast with ARIMA
# Set ARIMA parameters (p, d, q)
p, d, q = 5, 1, 0  # Adjust based on ACF/PACF plots or trial and error

history = [x for x in train]
predictions = []
test_dates = daily_sales['Date'][train_size:].reset_index(drop=True)

for t in range(len(test)):
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    history.append(test[t])  # Now this will work because test is a list

# Step 5: Evaluate the model
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f"Test RMSE: {rmse}")

# Step 6: Visualize the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(test_dates, test, label='Actual')
plt.plot(test_dates, predictions, label='Predicted', color='red')
plt.title('Rolling ARIMA Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

# Step 7: Predict future values (14/08/2024 - 28/08/2024) using rolling window
# Start with the full dataset and continue rolling forecast into the future
history = [x for x in daily_sales['total_sales']]  # Rebuild the full history
future_predictions = []
future_dates = pd.date_range(start="2024-08-14", end="2024-08-28", freq='D')

for future_date in future_dates:
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    future_predictions.append(yhat)
    history.append(yhat)  # Add the forecast to the history for the next iteration

# Step 8: Visualize the future predictions
plt.figure(figsize=(10, 6))
plt.plot(future_dates, future_predictions, label='Future Predicted', color='green')
plt.title('Future Sales Forecast (14/08/2024 - 28/08/2024)')
plt.xlabel('Date')
plt.ylabel('Predicted Total Sales')
plt.legend()
plt.show()

# Step 9: Display the future forecasted values
future_forecast = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Sales': future_predictions
})

print("\nFuture Sales Forecast (14/08/2024 - 28/08/2024):")
print(future_forecast)
