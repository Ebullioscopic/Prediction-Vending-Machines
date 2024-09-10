import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load and preprocess the data
file_path = "combined_file_updated.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Convert 'Timestamp' to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

# Aggregate daily sales (Total sales amount per day)
df['Date'] = df['Timestamp'].dt.date
daily_sales = df.groupby('Date').agg(total_sales=('Amount', 'sum')).reset_index()

# Step 2: Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
daily_sales['total_sales_scaled'] = scaler.fit_transform(daily_sales[['total_sales']])

# Step 3: Create sequences for LSTM (sliding window approach)
def create_sequences(data, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
        targets.append(data[i+window_size])
    return np.array(sequences), np.array(targets)

window_size = 10  # Look back 10 days to predict the next day's sales
data = daily_sales['total_sales_scaled'].values
X, y = create_sequences(data, window_size)

# Step 4: Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input data to 3D for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 5: Define and train the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(1))  # Output layer to predict the next day's sales

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Step 6: Evaluate the model on the test set
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)  # Inverse transform to get predictions in original scale
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  # Inverse transform the actual values

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE: {rmse}')

# Step 7: Visualize the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(daily_sales['Date'][train_size + window_size:], y_test, label='Actual Sales')
plt.plot(daily_sales['Date'][train_size + window_size:], y_pred, label='Predicted Sales', color='red')
plt.title('LSTM Rolling Forecast vs Actual Sales')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

# Step 8: Use the model to predict future values (14/08/2024 - 28/08/2024)
# We will use the last window of the training set to start predicting future sales

# Take the last window from the entire dataset to start predicting future sales
last_window = data[-window_size:].reshape((1, window_size, 1))

future_predictions = []
future_dates = pd.date_range(start="2024-08-14", end="2024-08-28", freq='D')

for future_date in future_dates:
    next_pred_scaled = model.predict(last_window)
    next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]  # Convert back to original scale
    future_predictions.append(next_pred)
    
    # Update the last window with the new prediction (rolling window)
    last_window = np.append(last_window[:, 1:, :], [[next_pred_scaled[0]]], axis=1)


# Step 9: Visualize the future predictions
plt.figure(figsize=(10, 6))
plt.plot(future_dates, future_predictions, label='Future Predicted Sales', color='green')
plt.title('Future Sales Forecast (14/08/2024 - 28/08/2024)')
plt.xlabel('Date')
plt.ylabel('Predicted Total Sales')
plt.legend()
plt.show()

# Step 10: Display the future forecasted values
future_forecast = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Sales': future_predictions
})

print("\nFuture Sales Forecast (14/08/2024 - 28/08/2024):")
print(future_forecast)
