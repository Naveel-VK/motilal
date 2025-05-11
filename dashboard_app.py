
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Step 1: Load preprocessed data
final_data = pd.read_csv("ind_nifty500list.csv")  # Original company data
cagr_volatility_df = pd.read_csv("nifty500_analysis.csv")  # Data with CAGR and Volatility

# Step 2: Convert 'Date' column to datetime format - CORRECTED
# Handle ISO8601 format with timezone
cagr_volatility_df['Date'] = pd.to_datetime(cagr_volatility_df['Date'], format='ISO8601')
# Remove timezone information
cagr_volatility_df['Date'] = cagr_volatility_df['Date'].dt.tz_localize(None)

# Step 3: Remove duplicate rows for companies with same name
cagr_volatility_df = cagr_volatility_df.drop_duplicates(subset=['Company Name'], keep='first')

# Step 4: Streamlit App Title
st.title("📊 Nifty 500 Stock Performance Dashboard")

# Sidebar for selecting company - sorted alphabetically
company = st.sidebar.selectbox("Select a company", sorted(final_data['Company Name'].unique()))

# Filter data for the selected company
company_data = cagr_volatility_df[cagr_volatility_df['Company Name'] == company]

# Show company details and stock price trend
st.subheader(f"📈 Stock Price Trend: {company}")

if not company_data.empty:
    # Create figure with proper sizing
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot actual stock prices
    ax.plot(company_data['Date'], company_data['Stock Price'], 
            label='Stock Price', linewidth=2, color='steelblue')
    
    # Format x-axis dates properly
    date_format = DateFormatter("%b-%Y")
    ax.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    
    # Format y-axis
    ax.set_ylabel('Stock Price (₹)')
    ax.set_title(f"Stock Price of {company}")
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 CAGR", f"{company_data['CAGR'].values[0] * 100:.2f}%")
    with col2:
        st.metric("📉 Volatility", f"{company_data['Volatility'].values[0] * 100:.2f}%")
else:
    st.warning("No data available for selected company")

# Summary Table for Top and Bottom Performers by CAGR
st.subheader("📋 Summary: Top & Bottom Performers")

# Get top and bottom performers
top_cagr = cagr_volatility_df.sort_values(by='CAGR', ascending=False).head(5)
bottom_cagr = cagr_volatility_df.sort_values(by='CAGR').head(5)

# Display in columns with improved formatting
col1, col2 = st.columns(2)
with col1:
    st.markdown("**🏆 Top 5 Performers (CAGR)**")
    st.dataframe(
        top_cagr[['Company Name', 'CAGR']]
        .style.format({"CAGR": "{:.2%}"})
        .background_gradient(cmap='Greens')
    )

with col2:
    st.markdown("**🚨 Bottom 5 Performers (CAGR)**")
    st.dataframe(
        bottom_cagr[['Company Name', 'CAGR']]
        .style.format({"CAGR": "{:.2%}"})
        .background_gradient(cmap='Reds')
    )
