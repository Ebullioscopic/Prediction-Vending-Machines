import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Step 1: Load the data
file_path = "combined_file_updated.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Convert 'Timestamp' to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

# Step 3: Aggregate sales per day (Prophet works best on daily or higher-level data)
# We'll group the data by day to create the `y` (target sales) and `ds` (date) columns required by Prophet
df['Date'] = df['Timestamp'].dt.date

# Aggregate daily sales (Total sales amount per day)
daily_sales = df.groupby('Date').agg(total_sales=('Amount', 'sum')).reset_index()

# Prepare the dataframe for Prophet (Prophet needs 'ds' as the date column and 'y' as the target variable)
prophet_df = daily_sales.rename(columns={'Date': 'ds', 'total_sales': 'y'})

# Step 4: Initialize the Prophet model
model = Prophet()

# Step 5: Fit the Prophet model to the historical data
model.fit(prophet_df)

# Step 6: Create a DataFrame to hold future dates for predictions (14/08/2024 to 28/08/2024)
# Generate future dates (15 days ahead)
future_dates = model.make_future_dataframe(periods=15, freq='D', include_history=False)

# Step 7: Predict future sales for the next 15 days (14/08/2024 to 28/08/2024)
forecast = model.predict(future_dates)

# Step 8: Visualize the predictions
# Prophet has built-in plot functions for visualizing the forecast and components
model.plot(forecast)
plt.title("Sales Forecast for 14/08/2024 to 28/08/2024")
plt.xlabel("Date")
plt.ylabel("Total Sales Amount")
plt.show()

# You can also plot the trend and seasonality components
model.plot_components(forecast)
plt.show()

# Step 9: Display the predicted sales for the future period
# The `yhat` column in the forecast contains the predicted sales
future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
print("\nFuture Sales Forecast (14/08/2024 - 28/08/2024):")
print(future_forecast)

# Step 10: Calculate aggregated sales per day
# Since Prophet gives daily forecasts, we already have daily sales forecasted in `yhat`
daily_sales_forecast = future_forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted_Amount'})

# Display daily sales predictions
print("\nDaily Sales Predictions:")
print(daily_sales_forecast)

# Step 11: Category-wise and machine-wise forecasting
# We can't directly get category-wise and machine-wise predictions using Prophet since it's an aggregated model
# However, you can use historical ratios to estimate these forecasts

# For illustration, we'll use ratios based on historical data to estimate category-wise and machine-wise sales
# Calculate the proportion of sales by product and machine in the historical data
product_sales_ratio = df.groupby('Product Name')['Amount'].sum() / df['Amount'].sum()
machine_sales_ratio = df.groupby('Machine ID')['Amount'].sum() / df['Amount'].sum()

# Multiply the daily sales forecast by these ratios to estimate future category and machine sales
category_sales_forecast = pd.DataFrame()
for product, ratio in product_sales_ratio.items():
    product_forecast = daily_sales_forecast.copy()
    product_forecast['Product Name'] = product
    product_forecast['Predicted_Amount'] *= ratio
    category_sales_forecast = pd.concat([category_sales_forecast, product_forecast])

# Display category-wise sales predictions
print("\nCategory-wise Sales Predictions:")
print(category_sales_forecast[['Date', 'Product Name', 'Predicted_Amount']])

# Machine-wise sales forecast
machine_sales_forecast = pd.DataFrame()
for machine, ratio in machine_sales_ratio.items():
    machine_forecast = daily_sales_forecast.copy()
    machine_forecast['Machine ID'] = machine
    machine_forecast['Predicted_Amount'] *= ratio
    machine_sales_forecast = pd.concat([machine_sales_forecast, machine_forecast])

# Display machine-wise sales predictions
print("\nMachine-wise Sales Predictions:")
print(machine_sales_forecast[['Date', 'Machine ID', 'Predicted_Amount']])
