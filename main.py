import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from forecasting import TimeSeriesForecasting

@st.cache_data
def fetch_data():
    df = pd.read_csv("Inflation Rates.csv")
    model = tf.keras.models.load_model("model.h5")
    return df, model


def preprocess_df(df):
    month_to_number = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].map(month_to_number).astype(str), format='%Y-%m')

    df.drop(['Year', 'Month'], axis=1, inplace=True)
    df.set_index('Date', inplace=True)
    df = df.sort_index(ascending=True)
    return df



def plot_climate_data(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["12-Month Inflation"],
                             mode='lines', name="Monthly Inflation", line=dict(color='salmon')))

    fig.update_layout(title='Kenyan Economy Inflation Rate Data',
                      xaxis_title='Year', yaxis_title='Monthly Inflation Rate(%)',
                      width=1000, height=500, showlegend=True)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    st.plotly_chart(fig)


def parse_data_from_dataframe(df):
    date = []
    inflation = []
    count = 0
    for rate in df['12-Month Inflation']:
        inflation_float = float(rate)
        inflation.append(inflation_float)
        date.append(int(count))
        count += 1
    return date, inflation


def get_window_and_series(df, window_size, split_time):
    dates, inflation = parse_data_from_dataframe(df)
    series = np.array(inflation)
    return window_size, series

def train_val_split(time, series, time_step):

    time_train = time[:time_step]
    series_train = series[:time_step]
    time_valid = time[time_step:]
    series_valid = series[time_step:]

    return time_valid

st.title('Kenyan Economy Inflation Rate Data')
df, model = fetch_data()
df = preprocess_df(df)
plot_climate_data(df)
window_size = 5
split_time = 156

date, inflation = parse_data_from_dataframe(df)

window_size, series = get_window_and_series(df, window_size, split_time)

time_valid = train_val_split(df.index, series, split_time)

forecasting = TimeSeriesForecasting(model, series, time_valid, window_size)

future_years = st.slider("Select Years into the Future for Forecasting", 3, 1, 36)
future_months = future_years * 12

with st.spinner("Forecasting..."):
    forecasting.plot_future_forecast(future_months)
