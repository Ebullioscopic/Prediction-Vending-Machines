# # Required Libraries
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime

# # Step 1: Load the data
# file_path = "combined_file_updated.csv"  # Update with your file path
# df = pd.read_csv(file_path)

# # Step 2: Preprocess the data
# # Convert 'Timestamp' to datetime
# #df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
# # Convert 'Timestamp' to datetime with 24-hour format
# #df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', dayfirst=True)
# df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
# # Extract time-based features
# df['Day'] = df['Timestamp'].dt.day
# df['Month'] = df['Timestamp'].dt.month
# df['Year'] = df['Timestamp'].dt.year
# df['Hour'] = df['Timestamp'].dt.hour
# df['Day_of_Week'] = df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6

# # Drop unnecessary columns (you can modify based on analysis)
# df.drop(columns=['Position'], inplace=True)

# # Step 3: Handle missing values (if any)
# df.fillna(0, inplace=True)

# # Step 4: Encode categorical variables
# label_encoder = LabelEncoder()

# df['Product Name'] = label_encoder.fit_transform(df['Product Name'])
# df['Payment Method'] = label_encoder.fit_transform(df['Payment Method'])
# df['Vend Status'] = label_encoder.fit_transform(df['Vend Status'])
# df['Machine Name'] = label_encoder.fit_transform(df['Machine Name'])

# # Step 5: Feature Engineering
# # Calculate total sales (Amount = Quantity * Unit Price)
# df['Amount'] = df['Quantity'] * df['Unit Price']

# # Aggregate sales by machine or date if needed
# df['Daily_Sales'] = df.groupby(df['Timestamp'].dt.date)['Amount'].transform('sum')
# df['Machine_Sales'] = df.groupby('Machine ID')['Amount'].transform('sum')

# # Step 6: Prepare for model training
# # Define the target variable (e.g., predicting 'Amount')
# target = 'Amount'

# # Select features (you can experiment with different combinations)
# features = ['Product Name', 'Quantity', 'Unit Price', 'Machine ID', 'Day', 'Month', 'Year', 'Hour', 'Day_of_Week', 'Machine_Sales']

# # Split the data
# X = df[features]
# y = df[target]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Step 7: Train the model
# model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
# model.fit(X_train, y_train)

# # Step 8: Evaluate the model
# y_pred = model.predict(X_test)

# # Calculate performance metrics
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# print(f"Mean Absolute Error: {mae}")
# print(f"Root Mean Squared Error: {rmse}")

# # Visualize predictions vs actual values
# plt.figure(figsize=(10, 6))
# plt.plot(y_test.values, label='Actual')
# plt.plot(y_pred, label='Predicted')
# plt.title('Actual vs Predicted Sales')
# plt.legend()
# plt.show()

# # Step 9: Predict future values (14/08/2024 - 28/08/2024)
# # Assuming you have similar data or structure for future timestamps
# future_dates = pd.date_range(start="2024-08-14", end="2024-08-28", freq='H')

# # Add necessary features as per the model's training data
# # These are placeholder values and should be adapted based on your context
# future_data = pd.DataFrame({
#     'Timestamp': future_dates,
#     'Day': future_dates.day,
#     'Month': future_dates.month,
#     'Year': future_dates.year,
#     'Hour': future_dates.hour,
#     'Day_of_Week': future_dates.dayofweek,
#     'Product Name': 0,  # Placeholder (use 0, median, or known value)
#     'Quantity': 1,      # Placeholder (use average or a realistic default value)
#     'Unit Price': 1.0,  # Placeholder (use average price or realistic value)
#     'Machine ID': 0,    # Placeholder (use known machine ID or default value)
#     'Machine_Sales': 0  # Fill with expected values or 0 for unknown machines
# })

# # Ensure all features are present as in X_train
# future_predictions = model.predict(future_data[features])
# future_data['Predicted_Amount'] = future_predictions

# # Show the prediction for the future period
# print(future_data[['Timestamp', 'Predicted_Amount']])

# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Step 1: Load the data
file_path = "combined_file_updated.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Convert 'Timestamp' to datetime with 24-hour format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

# Extract time-based features
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Year'] = df['Timestamp'].dt.year
df['Hour'] = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6

# Drop unnecessary columns (you can modify based on analysis)
df.drop(columns=['Position'], inplace=True)

# Step 3: Handle missing values (if any)
df.fillna(0, inplace=True)

# Step 4: Encode categorical variables
label_encoder = LabelEncoder()

df['Product Name'] = label_encoder.fit_transform(df['Product Name'])
df['Payment Method'] = label_encoder.fit_transform(df['Payment Method'])
df['Vend Status'] = label_encoder.fit_transform(df['Vend Status'])
df['Machine Name'] = label_encoder.fit_transform(df['Machine Name'])

# Step 5: Feature Engineering
# Calculate total sales (Amount = Quantity * Unit Price)
df['Amount'] = df['Quantity'] * df['Unit Price']

# Aggregate sales by machine or date if needed
df['Daily_Sales'] = df.groupby(df['Timestamp'].dt.date)['Amount'].transform('sum')
df['Machine_Sales'] = df.groupby('Machine ID')['Amount'].transform('sum')

# Step 6: Calculate most common placeholders
# Most common product
most_common_product = df['Product Name'].mode()[0]

# Most common Machine ID
most_common_machine_id = df['Machine ID'].mode()[0]

# Average Unit Price
average_unit_price = df['Unit Price'].mean()

# Most common quantity (in this case it's probably 1, but we calculate it)
most_common_quantity = df['Quantity'].mode()[0]

# Step 7: Prepare for model training
# Define the target variable (e.g., predicting 'Amount')
target = 'Amount'

# Select features (you can experiment with different combinations)
features = ['Product Name', 'Quantity', 'Unit Price', 'Machine ID', 'Day', 'Month', 'Year', 'Hour', 'Day_of_Week', 'Machine_Sales']

# Split the data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 8: Train the model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()

# # Step 10: Predict future values (14/08/2024 - 28/08/2024)
# # Generate future dates
# future_dates = pd.date_range(start="2024-08-14", end="2024-08-28", freq='h')

# # Add necessary features for future data, dynamically adjusting the placeholders
# future_data = pd.DataFrame({
#     'Timestamp': future_dates,
#     'Day': future_dates.day,
#     'Month': future_dates.month,
#     'Year': future_dates.year,
#     'Hour': future_dates.hour,
#     'Day_of_Week': future_dates.dayofweek,
#     'Product Name': most_common_product,  # Dynamically calculated most common product
#     'Quantity': most_common_quantity,     # Dynamically calculated most common quantity
#     'Unit Price': average_unit_price,     # Dynamically calculated average unit price
#     'Machine ID': most_common_machine_id, # Dynamically calculated most common machine ID
#     'Machine_Sales': 0                   # Initialize as 0 for now, as future sales are unknown
# })

# # Ensure all features are present as in X_train
# future_predictions = model.predict(future_data[features])
# future_data['Predicted_Amount'] = future_predictions

# # Show the prediction for the future period
# print(future_data[['Timestamp', 'Predicted_Amount']])

# Step 10: Predict future values (14/08/2024 - 28/08/2024)
# Generate future dates
future_dates = pd.date_range(start="2024-08-14", end="2024-08-28", freq='H')

# Introduce variability based on historical data
# For example, use the most common products but vary them by day or time

# Simulate Product Name variation (random sampling from top 5 most common products)
top_products = df['Product Name'].value_counts().index[:5]  # Top 5 most common products
future_product_names = np.random.choice(top_products, size=len(future_dates))

# Simulate Unit Price variation (sampling from the range of historical unit prices)
unit_price_min = df['Unit Price'].min()
unit_price_max = df['Unit Price'].max()
future_unit_prices = np.random.uniform(unit_price_min, unit_price_max, size=len(future_dates))

# Simulate Machine ID variation (randomly select from existing machines)
machine_ids = df['Machine ID'].unique()
future_machine_ids = np.random.choice(machine_ids, size=len(future_dates))

# Aggregate sales can be derived as rolling averages, here initialized as 0 for simplicity
# For more advanced usage, consider adding time-dependent rolling averages based on historical data
future_machine_sales = np.zeros(len(future_dates))

# Create future data with variability
future_data = pd.DataFrame({
    'Timestamp': future_dates,
    'Day': future_dates.day,
    'Month': future_dates.month,
    'Year': future_dates.year,
    'Hour': future_dates.hour,
    'Day_of_Week': future_dates.dayofweek,
    'Product Name': future_product_names,  # Vary Product Names
    'Quantity': most_common_quantity,                         # Quantity is still 1, but you can adjust this if needed
    'Unit Price': future_unit_prices,      # Vary Unit Prices
    'Machine ID': future_machine_ids,      # Vary Machine IDs
    'Machine_Sales': future_machine_sales  # Sales can be initialized to 0 or use historical rolling values
})

# Ensure all features are present as in X_train
future_predictions = model.predict(future_data[features])
future_data['Predicted_Amount'] = future_predictions

# Show the prediction for the future period
print(future_data[['Timestamp', 'Predicted_Amount']])

# Step 11: Aggregate predicted sales by day, category (product), and machine

# First, calculate total sales (each prediction represents a single sale)
future_data['Sales_Count'] = 1  # Each row represents a single sale

# Group by day and aggregate total sales and predicted amount
daily_sales = future_data.groupby(future_data['Timestamp'].dt.date).agg(
    total_sales=('Sales_Count', 'sum'),
    total_amount=('Predicted_Amount', 'sum')
).reset_index()

# Display daily sales
print("Daily Sales:")
print(daily_sales)

# Group by product (Product Name) and day
category_sales = future_data.groupby([future_data['Timestamp'].dt.date, 'Product Name']).agg(
    total_sales=('Sales_Count', 'sum'),
    total_amount=('Predicted_Amount', 'sum')
).reset_index()

# Display category-wise sales
print("\nCategory-wise Sales:")
print(category_sales)

# Group by machine (Machine ID) and day
machine_sales = future_data.groupby([future_data['Timestamp'].dt.date, 'Machine ID']).agg(
    total_sales=('Sales_Count', 'sum'),
    total_amount=('Predicted_Amount', 'sum')
).reset_index()

# Display machine-wise sales
print("\nMachine-wise Sales:")
print(machine_sales)
