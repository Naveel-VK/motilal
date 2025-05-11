# Install necessary packages
!pip install yfinance plotly streamlit pyngrok --quiet

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from google.colab import files
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
from pyngrok import ngrok

# Step 1: Upload your 'ind_nifty500list.csv' file manually
uploaded = files.upload()

# Load the uploaded CSV file
df = pd.read_csv('ind_nifty500list.csv')

# Display the first few rows to confirm
print("Initial Data Preview:")
print(df.head())

# Check for missing values
print("\nMissing Values Check:")
print(df.isnull().sum())

# Step 2: Fetch stock data for Nifty 500 companies
start_date = "2010-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

all_data = pd.DataFrame()

for symbol in df['Symbol']:
    try:
        print(f"Fetching data for {symbol}...")
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(start=start_date, end=end_date)

        if not hist.empty:
            hist['Company Name'] = symbol
            hist = hist[['Close', 'Dividends', 'Company Name']]
            hist.reset_index(inplace=True)
            all_data = pd.concat([all_data, hist], ignore_index=True)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

# Rename columns for consistency
all_data.rename(columns={'Close': 'Stock Price', 'Dividends': 'Dividend'}, inplace=True)

# Step 3: Feature Engineering - Calculate Total Return
all_data['Total Return'] = all_data['Stock Price'] + all_data['Dividend']

# Calculate CAGR (Compound Annual Growth Rate)
def calculate_cagr(start_price, end_price, years):
    return (end_price / start_price) ** (1 / years) - 1

cagr_dict = {}
for symbol in df['Symbol']:
    stock_data = all_data[all_data['Company Name'] == symbol]
    if stock_data.empty:
        print(f"No data available for {symbol}, skipping...")
        continue

    start_price = stock_data.iloc[0]['Stock Price']
    end_price = stock_data.iloc[-1]['Stock Price']
    years = (stock_data['Date'].iloc[-1] - stock_data['Date'].iloc[0]).days / 365
    cagr_dict[symbol] = calculate_cagr(start_price, end_price, years)

cagr_df = pd.DataFrame(list(cagr_dict.items()), columns=['Company Name', 'CAGR'])

# Step 4: Calculate Volatility (Annualized Standard Deviation of Daily Returns)
all_data['Daily Returns'] = all_data['Stock Price'].pct_change()
volatility = all_data.groupby('Company Name')['Daily Returns'].std() * (252 ** 0.5)

volatility_df = volatility.reset_index()
volatility_df.rename(columns={'Daily Returns': 'Volatility'}, inplace=True)

# Step 5: Merge the CAGR and Volatility data into the original data
final_data = pd.merge(all_data, cagr_df, on='Company Name', how='left')
final_data = pd.merge(final_data, volatility_df, on='Company Name', how='left')

# Save the final data to a CSV file
final_data.to_csv("nifty500_analysis.csv", index=False)
print("✅ All data fetched, analyzed, and saved!")

# Step 6: Data Visualization
# Plot Stock Price of a single company (Example: Reliance)
symbol_data = all_data[all_data['Company Name'] == 'RPOWER']
plt.plot(symbol_data['Date'], symbol_data['Stock Price'])
plt.title('Reliance Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

# Visualize the CAGR Distribution
plt.hist(cagr_df['CAGR'], bins=50)
plt.title('CAGR Distribution of Nifty 500 Stocks')
plt.xlabel('CAGR')
plt.ylabel('Frequency')
plt.show()

# Top 10 Companies by CAGR
top_cagr = cagr_df.sort_values(by='CAGR', ascending=False).head(10)
plt.figure(figsize=(10, 5))
plt.bar(top_cagr['Company Name'], top_cagr['CAGR'] * 100)
plt.xticks(rotation=45)
plt.title('Top 10 Companies by CAGR (%)')
plt.ylabel('CAGR (%)')
plt.tight_layout()
plt.show()

# Most Volatile Companies
most_volatile = volatility_df.sort_values(by='Volatility', ascending=False).head(10)
plt.figure(figsize=(10, 5))
plt.bar(most_volatile['Company Name'], most_volatile['Volatility'] * 100)
plt.xticks(rotation=45)
plt.title('Most Volatile Nifty 500 Stocks')
plt.ylabel('Annualized Volatility (%)')
plt.tight_layout()
plt.show()

# Step 7: Create a Streamlit Dashboard
code = '''
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load preprocessed data
final_data = pd.read_csv("nifty500_analysis.csv")
cagr_volatility_df = pd.read_csv("nifty500_cagr_volatility.csv")

st.title("📊 Nifty 500 Stock Performance Dashboard")

# Sidebar filters
company = st.sidebar.selectbox("Select a company", final_data['Company Name'].unique())

# Show company details
company_data = final_data[final_data['Company Name'] == company]

st.subheader(f"📈 Stock Price Trend: {company}")
fig, ax = plt.subplots()
ax.plot(company_data['Date'], company_data['Stock Price'], label='Stock Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
st.pyplot(fig)

# Show CAGR & Volatility
cagr_row = cagr_volatility_df[cagr_volatility_df['Company Name'] == company]
if not cagr_row.empty:
    st.metric("📈 CAGR (%)", f"{cagr_row['CAGR'].values[0] * 100:.2f}")
    st.metric("📉 Volatility (%)", f"{cagr_row['Volatility'].values[0] * 100:.2f}")

# Summary Table
st.subheader("📋 Summary: Top & Bottom Performers")
top_cagr = cagr_volatility_df.sort_values(by='CAGR', ascending=False).head(5)
bottom_cagr = cagr_volatility_df.sort_values(by='CAGR').head(5)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**🏆 Top 5 CAGR Performers**")
    st.dataframe(top_cagr[['Company Name', 'CAGR']].style.format({"CAGR": "{:.2%}"}))

with col2:
    st.markdown("**🚨 Bottom 5 CAGR Performers**")
    st.dataframe(bottom_cagr[['Company Name', 'CAGR']].style.format({"CAGR": "{:.2%}"}))
'''

# Write the Streamlit app to a Python file
with open("dashboard_app.py", "w") as f:
    f.write(code)

# Step 8: Expose the Streamlit app via ngrok
ngrok.set_auth_token("2vnidcStEtI9beu2PzWGLwjg4UN_5uHcwThJDeh2wrzVCtTKp")
public_url = ngrok.connect("http://localhost:8501")
print("Streamlit App URL:", public_url)

# Run Streamlit app
!streamlit run dashboard_app.py &> streamlit_log.txt &
