import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Muat model ARIMA dari file .sav
model_file = 'model_forecast-ar.sav'
model = pickle.load(open(model_file, 'rb'))

# Muat dataset AirPassengers.csv
data_file = 'AirPassengers.csv'
data = pd.read_csv(data_file)
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Judul aplikasi
st.title('ARIMA Forecasting App')

# Slider untuk menentukan jumlah bulan yang akan diprediksi
forecast_steps = st.slider('Jumlah Bulan Prediksi', 1, 24, 12)

# Tombol "Prediksi"
if st.button('Prediksi'):
    # Prediksi dengan model ARIMA
    forecast, stderr, conf_int = model.forecast(steps=forecast_steps)

    # Dekomposisi musiman
    result = seasonal_decompose(data, model='additive')

    # Tampilkan data asli
    st.subheader('Data Asli')
    st.line_chart(data)

    # Tampilkan hasil dekomposisi
    st.subheader('Hasil Dekomposisi')
    st.line_chart(result.trend)
    st.line_chart(result.seasonal)
    st.line_chart(result.resid)

    # Tampilkan hasil prediksi
    st.subheader('Hasil Prediksi')
    st.line_chart(forecast)

    # Tampilkan tabel hasil prediksi
    st.subheader('Tabel Hasil Prediksi')
    forecast_df = pd.DataFrame({
        'Tanggal': pd.date_range(start=data.index[-1], periods=forecast_steps + 1, closed='right'),
        'Prediksi': forecast
    })
    st.dataframe(forecast_df)
