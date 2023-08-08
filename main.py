import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from forecasting import TimeSeriesForecasting


def fetch_data():
    df = pd.read_csv("Inflation Rates.csv")
    model = tf.keras.models.load_model("model.h5")
    return df, model


def preprocess_dataframe(df):
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


def get_window_and_series(df, split_time):
    dates, inflation = parse_data_from_dataframe(df)
    series = np.array(inflation)
    return  series

def train_val_split(time, series, time_step):

    time_train = time[:time_step]
    series_train = series[:time_step]
    time_valid = time[time_step:]
    series_valid = series[time_step:]

    return time_train, series_train, time_valid, series_valid

def future_model_forecast(model, series, window_size, future_steps):
    # Prepare the dataset for prediction
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    dataset = dataset.batch(5).prefetch(1)
    forecast = model.predict(dataset)
    future_forecast = []
    for batch in forecast:
        future_forecast.extend(batch[-future_steps:])
    return np.array(future_forecast).squeeze()


def plot_future_forecast(model, series, time_valid, window_size, future_months):
    last_timestamp = time_valid[-1]
    future_time_steps = future_months
    future_time = pd.date_range(start=last_timestamp, periods=future_time_steps, freq='1M')
    future_forecast = future_model_forecast(model, series, window_size, future_time_steps)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=time_valid, y=series, mode='lines', name='Actual Data', line=dict(color='salmon')))

    fig.add_trace(go.Scattergl(x=future_time, y=future_forecast, mode='lines', name='Predicted Data (Future)', line=dict(color='green')))

    fig.update_layout(title='Actual vs. Predicted Data', xaxis_title='Time', yaxis_title='Value')
    st.plotly_chart(fig)

    
def main():
    st.title('Time Series Prediction on Inflation Rate Data in Kenya')
    df, model = fetch_data()
    df = preprocess_dataframe(df)
    plot_climate_data(df)

    window_size = 5
    split_time = 156
    series = get_window_and_series(df, split_time)

    time_train, series_trainset, time_valid, series_validset = train_val_split(df.index, series, split_time)

    # forecasting = TimeSeriesForecasting(model, series, time_valid, window_size)

    future_years = st.slider("Select Month into the Future for Forecasting", 3, 1, 36)
    future_months = future_years * 12

    with st.spinner("Forecasting..."):
        plot_future_forecast(model, series, time_valid, window_size, future_months)

if __name__ == "__main__":
    main()