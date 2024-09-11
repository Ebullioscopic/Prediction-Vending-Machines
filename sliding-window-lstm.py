import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load the data
file_path = "combined_file_updated.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Convert 'Timestamp' to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

# Step 2: Group by Machine, Product, and Date
df['Date'] = df['Timestamp'].dt.date
daily_sales = df.groupby(['Machine Name', 'Product Name', 'Date']).agg(total_sales=('Amount', 'sum')).reset_index()

# Get a list of machines and products
machines = daily_sales['Machine Name'].unique()
products = daily_sales['Product Name'].unique()

# Prepare final output DataFrame
output_forecast = pd.DataFrame(columns=['Date', 'Machine Name', 'Product Name', 'Actual Sale', 'Forecasted Sale', 'Difference', 'Difference %'])

# Step 3: Build the LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Step 4: Train and Predict for each Machine and Product combination
for machine in machines:
    for product in products:
        # Filter data for this machine and product combination
        product_data = daily_sales[(daily_sales['Machine Name'] == machine) & (daily_sales['Product Name'] == product)]
        
        if len(product_data) < 20:  # Skip if insufficient data for this machine-product combination
            continue

        # Scale the sales data
        scaler = MinMaxScaler(feature_range=(0, 1))
        product_data['total_sales_scaled'] = scaler.fit_transform(product_data[['total_sales']])

        # Create sequences (sliding window approach)
        def create_sequences(data, window_size):
            sequences = []
            targets = []
            for i in range(len(data) - window_size):
                sequences.append(data[i:i+window_size])
                targets.append(data[i+window_size])
            return np.array(sequences), np.array(targets)

        window_size = 10  # Use a window of 10 days
        data = product_data['total_sales_scaled'].values
        X, y = create_sequences(data, window_size)

        # Split into training and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Reshape for LSTM (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build and train the LSTM model
        model = build_lstm_model((window_size, 1))
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        # Step 5: Predict on the test set and calculate the difference
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        test_dates = product_data['Date'].iloc[train_size+window_size:].reset_index(drop=True)
        forecasted_sales = pd.DataFrame({
            'Date': test_dates,
            'Machine Name': machine,
            'Product Name': product,
            'Actual Sale': y_test.flatten(),
            'Forecasted Sale': y_pred.flatten()
        })
        forecasted_sales['Difference'] = forecasted_sales['Actual Sale'] - forecasted_sales['Forecasted Sale']
        forecasted_sales['Difference %'] = (forecasted_sales['Difference'] / forecasted_sales['Actual Sale']) * 100

        output_forecast = pd.concat([output_forecast, forecasted_sales])

        # Step 6: Predict future values (14/08/2024 to 28/08/2024)
        last_window = data[-window_size:].reshape((1, window_size, 1))
        future_predictions = []
        future_dates = pd.date_range(start="2024-08-14", end="2024-08-28", freq='D')

        for future_date in future_dates:
            next_pred_scaled = model.predict(last_window)
            next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]
            future_predictions.append(next_pred)
            last_window = np.append(last_window[:, 1:, :], [[next_pred_scaled[0]]], axis=1)

        future_forecast = pd.DataFrame({
            'Date': future_dates,
            'Machine Name': machine,
            'Product Name': product,
            'Actual Sale': [None] * len(future_dates),  # No actual sales data for the future
            'Forecasted Sale': future_predictions,
            'Difference': [None] * len(future_dates),
            'Difference %': [None] * len(future_dates)
        })

        output_forecast = pd.concat([output_forecast, future_forecast])

# Step 7: Final Output
output_forecast.reset_index(drop=True, inplace=True)
print(output_forecast)

# Save the output to CSV if needed
output_forecast.to_csv('forecasted_inventory.csv', index=False)

# Step 8: Visualize for a specific machine and product (optional)
specific_machine = machines[0]
specific_product = products[0]
specific_forecast = output_forecast[(output_forecast['Machine Name'] == specific_machine) & (output_forecast['Product Name'] == specific_product)]

plt.figure(figsize=(10, 6))
plt.plot(specific_forecast['Date'], specific_forecast['Actual Sale'], label='Actual Sales')
plt.plot(specific_forecast['Date'], specific_forecast['Forecasted Sale'], label='Forecasted Sales', color='red')
plt.title(f'Sales Forecast for {specific_machine} - {specific_product}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.xticks(rotation=45)
plt.show()
