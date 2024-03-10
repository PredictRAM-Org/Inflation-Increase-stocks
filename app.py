import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import os

# Load inflation data
inflation_data = pd.read_excel('inflation.xlsx')

# Load stock data from the folder
stock_folder_path = 'stockfolder'
stock_files = os.listdir(stock_folder_path)

# Display app title and description
st.title('Stock Analysis App')
st.write('Analyze monthly stock price returns and compare them with inflation changes.')

# Sidebar for user input
selected_month = st.sidebar.selectbox('Select Month:', inflation_data['Month'].unique())
inflation_index = inflation_data[inflation_data['Month'] == selected_month]['Inflation Index'].values[0]

# Load and preprocess stock data
stock_data = pd.DataFrame()
for file in stock_files:
    try:
        stock = pd.read_csv(os.path.join(stock_folder_path, file), encoding='utf-8')
        
        # Convert date to 'YYYY-MM-DD' format
        stock['Date'] = pd.to_datetime(stock['Date']).dt.strftime('%Y-%m-%d')
        
        stock_data = pd.concat([stock_data, stock], ignore_index=True)
    except pd.errors.ParserError as pe:
        st.warning(f"Error reading file {file}: {pe}")
        st.warning(f"Skipping file {file} due to parsing error.")
        continue
    except Exception as e:
        st.warning(f"Error reading file {file}: {e}")

# Merge inflation data with stock data on the 'Date' column
merged_data = pd.merge(stock_data, inflation_data, how='inner', left_on='Date', right_on='Month')

# Feature engineering: Create lag features for inflation index
merged_data['Inflation Index Lag'] = merged_data['Inflation Index'].shift(1)
merged_data = merged_data.dropna()

# Calculate monthly stock price returns
merged_data['Stock Return'] = merged_data.groupby('Stock')['Close'].pct_change() * 100

# Split data into train and test sets
train_size = int(0.8 * len(merged_data))
train_data, test_data = merged_data[:train_size], merged_data[train_size:]

# Select features and target variable
features = ['Open', 'High', 'Low', 'Close', 'Inflation Index Lag']
target = 'Stock Return'

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(train_data[features], train_data[target])

# Train Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(train_data[features], train_data[target])

# Make predictions
rf_predictions = rf_model.predict(test_data[features])
dt_predictions = dt_model.predict(test_data[features])

# Calculate RMSE for each model
rf_rmse = mean_squared_error(test_data[target], rf_predictions, squared=False)
dt_rmse = mean_squared_error(test_data[target], dt_predictions, squared=False)

# Display results
st.subheader('Model Comparison:')
st.write('Random Forest RMSE:', rf_rmse)
st.write('Decision Tree RMSE:', dt_rmse)

# Display stock with the highest increase when inflation increased
st.subheader('Stock with the Highest Increase:')
stock_with_highest_increase = test_data.loc[test_data['Stock Return'].idxmax()]['Stock']
st.write(f'The stock with the highest increase in monthly return is: {stock_with_highest_increase}')

# Display comparison of stock returns and inflation changes
st.subheader('Monthly Stock Returns vs. Inflation Changes:')
comparison_data = test_data[['Date', 'Stock', 'Stock Return', 'Inflation Index Lag']]
st.line_chart(comparison_data.set_index('Date'))
