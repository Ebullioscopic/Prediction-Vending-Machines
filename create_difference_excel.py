import pandas as pd

# Step 1: Load the data from the Excel file
df = pd.read_excel('forecast_lstm_max_rollingwindow.xlsx')

# Step 2: Convert 'AbsDate' to datetime
df['AbsDate'] = pd.to_datetime(df['AbsDate'], format='%d/%m/%y', errors='coerce')

# Step 3: Filter the DataFrame to include only the forecast period (14/08/2024 to 28/08/2024)
forecast_period = df[(df['AbsDate'] >= '2024-08-14') & (df['AbsDate'] <= '2024-08-28')]

# Step 4: Create a pivot table to sum the Forecasted Sale per day (AbsDate)
pivot_table = forecast_period.pivot_table(
    index=['AbsDate'],              # Group by Date (AbsDate)
    values='Forecasted Sale',       # Sum the Forecasted Sale per day
    aggfunc='sum'                   # Use sum function to aggregate sales
)

# Convert the pivot table to a DataFrame for further processing
forecasted_sales_by_day = pivot_table.reset_index()

# Step 5: Actual sales data as a dictionary (from the query)
actual_sales_data = {
    '2024-08-14': 107855,
    '2024-08-15': 74050,
    '2024-08-16': 85539,
    '2024-08-17': 81800,
    '2024-08-18': 63440,
    '2024-08-19': 74175,
    '2024-08-20': 97695,
    '2024-08-21': 101910,
    '2024-08-22': 118040,
    '2024-08-23': 110575,
    '2024-08-24': 84975,
    '2024-08-25': 73596,
    '2024-08-26': 65071,
    '2024-08-27': 87989,
    '2024-08-28': 79241
}

# Step 6: Calculate the percentage difference and average it
total_difference_percentage = 0
total_days = 0

# Loop through the forecasted data and calculate differences
for index, row in forecasted_sales_by_day.iterrows():
    date_str = row['AbsDate'].strftime('%Y-%m-%d')  # Convert date to string to match keys in actual_sales_data
    if date_str in actual_sales_data:
        actual_sales = actual_sales_data[date_str]
        forecasted_sales = row['Forecasted Sale']

        # Calculate percentage difference for the day
        difference = abs((actual_sales - forecasted_sales)/10) / actual_sales * 100
        total_difference_percentage += difference
        total_days += 1

# Step 7: Calculate average difference percentage (modulus sum / total count)
if total_days > 0:
    average_difference_percentage = total_difference_percentage / total_days
else:
    average_difference_percentage = 0

# Step 8: Output the average difference percentage
print(f"Average Difference Percentage: {average_difference_percentage:.2f}%")
