import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import log, sqrt, exp

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject into Streamlit
st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    width: auto;
    margin: 0 auto;
}

.metric-call {
    background-color: #90ee90;
    color: black;
    margin-right: 10px;
    border-radius: 10px;
}

.metric-put {
    background-color: #ffcccb;
    color: black;
    border-radius: 10px;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0;
}

.metric-label {
    font-size: 1rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (log(current_price / strike) + (interest_rate + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * sqrt(time_to_maturity))
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        put_price = strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)
        self.call_gamma = norm.pdf(d1) / (strike * volatility * sqrt(time_to_maturity))
        self.put_gamma = self.call_gamma

        return call_price, put_price

# Function to generate heatmaps
def plot_heatmap(bs_model, spot_range, vol_range, strike, option_type):
    price_diffs = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            predicted_price = bs_temp.call_price if option_type == "call" else bs_temp.put_price
            price_diffs[i, j] = predicted_price
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(price_diffs, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(f'{option_type.upper()} Price Heatmap')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Volatility')
    
    return fig

# Function to generate color maps for price differences
def plot_price_diff_colormap(bs_model, spot_range, vol_range, purchase_price_range, option_type):
    price_diffs = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=bs_model.strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            predicted_price = bs_temp.call_price if option_type == "call" else bs_temp.put_price
            price_diffs[i, j] = predicted_price - purchase_price_range[i]  # Adjusted for varying purchase prices
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(10, 150, as_cmap=True)  # Custom colormap from bright green (negative) to bright red (positive)
    sns.heatmap(price_diffs, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap=cmap, ax=ax)
    ax.set_title(f'{option_type.upper()} Price Difference')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Volatility')
    
    return fig


# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://in.linkedin.com/in/raghav-srivastava-11001ai"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Raghav Srivastava`</a>', unsafe_allow_html=True)

    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
    purchase_min_call = st.number_input('Min Purchase Price for Call', min_value=0.01, value=10.0*0.8, step=0.01)
    purchase_max_call = st.number_input('Max Purchase Price for Call', min_value=0.01, value=10.0*1.2, step=0.01)
    purchase_min_put = st.number_input('Min Purchase Price for Put', min_value=0.01, value=10.0*0.8, step=0.01)
    purchase_max_put = st.number_input('Max Purchase Price for Put', min_value=0.01, value=10.0*1.2, step=0.01)

    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)
    purchase_call_range = np.linspace(purchase_min_call, purchase_max_call, 10)
    purchase_put_range = np.linspace(purchase_min_put, purchase_max_put, 10)

# Main Page for Output Display
st.title("Black-Scholes Pricing Model")

# Table of Inputs
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate Call and Put values
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call = plot_heatmap(bs_model, spot_range, vol_range, strike, "call")
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Price Heatmap")
    heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike, "put")
    st.pyplot(heatmap_fig_put)

# Color maps for the difference between predicted and purchase prices
st.title("Options Price Difference - Interactive Color Map")
st.info("Explore the differences between predicted option prices and varying purchase prices.")

col1, col2 = st.columns([1, 1], gap="small")

with col1:
    st.subheader("Call Price Difference Color Map")
    call_diff_colormap = plot_price_diff_colormap(bs_model, spot_range, vol_range, purchase_call_range, "call")
    st.pyplot(call_diff_colormap)

with col2:
    st.subheader("Put Price Difference Color Map")
    put_diff_colormap = plot_price_diff_colormap(bs_model, spot_range, vol_range, purchase_put_range, "put")
    st.pyplot(put_diff_colormap)
