import pandas as pd
import numpy as np
from pmdarima import auto_arima
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv('/content/combined_file_updated.csv')

# Convert the Timestamp column to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%Y-%m-%d %H:%M:%S')

# Sort the data by machine and timestamp
data = data.sort_values(by=['Machine Name', 'Timestamp'])

# Filter out the first 5 days for each machine
data['Day_Count'] = data.groupby('Machine Name').cumcount() + 1
filtered_data = data[data['Day_Count'] > 5]

# Assign region based on machine name
filtered_data['Region'] = np.where(filtered_data['Machine Name'].str.startswith('GG'), 'NCR', 'Bhopal')

# Select relevant columns for prediction
filtered_data = filtered_data[['Timestamp', 'Machine Name', 'Region', 'Amount', 'Product Name', 'Machine ID', 'Unit Price']]

# Function to forecast sales for each product within each machine
def forecast_sales(data):
    forecast_list = []
    
    # Group the data by machine and product
    grouped_data = data.groupby(['Machine Name', 'Product Name'])

    for (machine_name, product_name), group in grouped_data:
        if len(group) < 10:
            # Skip if there's insufficient data
            print(f"Skipping {machine_name} - {product_name}: Not enough data (only {len(group)} records)")
            continue

        # Group by date to get daily total sales
        daily_sales = group.groupby('Timestamp').sum().reset_index()

        # Prepare the data for ARIMA
        daily_sales.set_index('Timestamp', inplace=True)

        try:
            # Train the ARIMA model
            model = auto_arima(daily_sales['Amount'], seasonal=True, m=7)
            # Forecast the next 7 days
            forecast, confidence_intervals = model.predict(n_periods=7, return_conf_int=True)

            # Create a forecast dataframe
            future_dates = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), periods=7)
            forecast_df = pd.DataFrame({
                'date': future_dates, 
                'forecasted_sales': forecast, 
                'lower_ci': confidence_intervals[:, 0], 
                'upper_ci': confidence_intervals[:, 1],
                'Machine ID': group['Machine ID'].iloc[0],
                'Product Name': product_name,
                'forecasted_amount': forecast * group['Unit Price'].iloc[0]
            })

            forecast_list.append(forecast_df)

        except ValueError as e:
            print(f"ARIMA model failed for {machine_name} - {product_name}: {e}")
            continue
        
    # Combine the forecasts for all products and machines
    combined_forecast = pd.concat(forecast_list, ignore_index=True)
    return combined_forecast

# Perform the forecasting
combined_forecast = forecast_sales(filtered_data)

# Sort by date for readability
combined_forecast.sort_values(by='date', inplace=True)

# Check if combined_forecast is empty
if combined_forecast.empty:
    print("Combined forecast is empty. Please check if there's sufficient data.")
else:
    # Print the forecast
    print(combined_forecast)

    # Save the forecast to CSV
    combined_forecast.to_csv('forecast.csv', index=False)

    # Save model parameters (example for one product)
    with open('model_params.pkl', 'wb') as f:
        pickle.dump({'Machine Name': 'example_machine', 'Product Name': 'example_product'}, f)
