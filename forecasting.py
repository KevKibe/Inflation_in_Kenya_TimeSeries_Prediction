import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import streamlit as st




class TimeSeriesForecasting:
    def __init__(self, model, series, time_valid, window_size):
        self.model = model
        self.series = series
        self.time_valid = time_valid
        self.window_size = window_size
    
    def future_model_forecast(self, future_steps):
        dataset = tf.data.Dataset.from_tensor_slices(self.series)
        dataset = dataset.window(self.window_size, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda w: w.batch(self.window_size))
        dataset = dataset.batch(5).prefetch(1)
        
        forecast = self.model.predict(dataset)
        
        future_forecast = []
        for batch in forecast:
            future_forecast.extend(batch[-future_steps:])
        
        return np.array(future_forecast).squeeze()
    
    def plot_future_forecast(self, future_months):
        last_timestamp = self.time_valid[-1]
        future_time_steps = future_months
        future_time = pd.date_range(start=last_timestamp, periods=future_time_steps, freq='1M')
        future_forecast = self.future_model_forecast(future_time_steps)
        
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=self.time_valid, y=self.series, mode='lines', name='Actual Data', line=dict(color='salmon')))
        fig.add_trace(go.Scattergl(x=future_time, y=future_forecast, mode='lines', name='Predicted Data (Future)', line=dict(color='green')))
        
        fig.update_layout(title='Actual vs. Predicted Data', xaxis_title='Time', yaxis_title='Value')
        st.plotly_chart(fig)




