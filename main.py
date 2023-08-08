import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

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



# def future_model_forecast(model, series, window_size, future_steps):
#     # Prepare the dataset for prediction
#     dataset = tf.data.Dataset.from_tensor_slices(series)
#     dataset = dataset.window(window_size, shift=1, drop_remainder=True)
#     dataset = dataset.flat_map(lambda w: w.batch(window_size))
#     dataset = dataset.batch(5).prefetch(1)

#     # Predict using the model
#     forecast = model.predict(dataset)

#     # Extract predictions for future time steps
#     future_forecast = []

#     for batch in forecast:
#         future_forecast.extend(batch[-future_steps:])

#     return np.array(future_forecast).squeeze()


# def plot_future_forecast(model, series, time_valid, window_size, future_months):
#     # Get the last timestamp from the validation time series
#     last_timestamp = time_valid[-1]
#     future_time_steps = future_months
#     # Generate a range of future time steps on a monthly basis
#     future_time = pd.date_range(start=last_timestamp, periods=future_months, freq='1M')

#     # Get the future forecast using the provided function future_model_forecast
#     future_forecast = future_model_forecast(model, series, window_size, future_months)

#     # Create a Plotly figure
#     fig = go.Figure()

#     # Add a trace for the actual data
#     fig.add_trace(go.Scattergl(x=time_valid, y=series, mode='lines', name='Actual Data', line=dict(color='salmon')))

#     # Add a trace for the predicted data (future forecast)
#     fig.add_trace(go.Scattergl(x=future_time, y=future_forecast, mode='lines', name='Predicted Data (Future)', line=dict(color='green')))

#     # Customize the layout of the figure
#     fig.update_layout(title='Actual vs. Predicted Data', xaxis_title='Time', yaxis_title='Value')

#     # Display the figure
#     fig.show()

st.title('Kenyan Economy Inflation Rate Data')
df, model = fetch_data()
df = preprocess_df(df)
    
plot_climate_data(df)

