# Inflation_in_Kenya_TimeSeries_Prediction
## Description
- This is a time-series forecasting application used for predicting future inflation rate in Kenya.
- This [notebook](https://github.com/KevKibe/Inflation_in_Kenya_TimeSeries_Prediction/blob/main/Timeseries_Inflation_Rate.ipynb) shows how I trained the model
  
## Dataset
The dataset is from this [link](https://www.centralbank.go.ke/inflation-rates/)https://www.centralbank.go.ke/inflation-rates/.
The columns in the dataset include:
- 12-month inflation: normally considered as inflation rate, is defined as the percentage change in the monthly consumer price index (CPI). For example, the 12-month inflation rate for November 2017 is the percentage change in the CPI of November 2017 and November 2016.
-  Annual average inflation: is the percentage change in the annual average consumer price index (CPI) of the corresponding months e.g. November 2017 and November 2016.
- Source: Kenya National Bureau of Statistics.

## Model
-This is the model consists of a 1D convolutional layer, followed by three LSTM (Long Short-Term Memory) layers, and three dense layers, with a single unit output layer for prediction.
   
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1,
                               activation='relu', input_shape=[window_size,1]),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

## Results
- The model achieved a Mean Squared Error of `1.28` and Mean Absolute Error of `0.89`.
  
**A plot of the predicted vs the actual values on the Validation Set**
- orange is predicted value, blue is actual value.
  
  ![image](https://github.com/KevKibe/Inflation_in_Kenya_TimeSeries_Prediction/assets/86055894/2a0b0352-7569-420a-b675-c0ee1f29283b)

**:zap: I'm currently open for roles in Data Science, Machine Learning, NLP and Computer Vision.**

