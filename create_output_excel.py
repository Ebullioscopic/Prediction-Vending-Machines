import pandas as pd

# Step 1: Load the data from the Excel file
df = pd.read_excel('forecast_lstm_max_rollingwindow.xlsx')

# Step 2: Convert 'AbsDate' to datetime
df['AbsDate'] = pd.to_datetime(df['AbsDate'], format='%d/%m/%y', errors='coerce')

# Step 3: Filter the DataFrame to include only the forecast period (14/08/2024 to 28/08/2024)
forecast_period = df[(df['AbsDate'] >= '2024-08-14') & (df['AbsDate'] <= '2024-08-28')]

# Step 4: Create a pivot table to sum the Forecasted Sale, grouped by AbsDate, Machine Name, and Product Name
pivot_table = forecast_period.pivot_table(
    index=['AbsDate', 'Machine Name', 'Product Name'],  # Group by Date (AbsDate), Machine Name, and Product Name
    values='Forecasted Sale',                          # Sum the Forecasted Sale
    aggfunc='sum'                                      # Use sum function to aggregate sales
)

# Step 5: Display the resulting pivot table
print(pivot_table)

# Step 6: Save the pivot table to an Excel file (optional)
pivot_table.to_excel('pivot_table_forecasted_sales_by_day_machine_product.xlsx', sheet_name='Forecast_Summary')
