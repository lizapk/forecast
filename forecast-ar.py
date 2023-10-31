import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# Muat model ARIMA dari file .sav
model_file = 'forecast-ar.sav'
model = pickle.load(open(model_file, 'rb'))

# Muat dataset AirPassengers.csv
data_file = 'AirPassengers.csv'
data = pd.read_csv(data_file)
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Judul aplikasi
st.title('ARIMA Forecasting App')

# Slider untuk menentukan jumlah bulan yang akan diprediksi
forecast_steps = st.slider('Jumlah Bulan Prediksi', 1, 36, 12)

# Tombol "Prediksi"
if st.button('Prediksi'):
    # Prediksi dengan model ARIMA
    forecast = model.forecast(steps=forecast_steps)

    col1, col2 = st.columns([2,3])
    with col1:
        # Tampilkan data asli
        st.subheader('Data Asli')
        st.line_chart(data)
    with col2:
        # Tampilkan hasil prediksi
        st.subheader('Hasil Prediksi')
        st.line_chart(forecast)
        
    col1, col2 = st.columns([2,3])
    with col1:
        # Tampilkan grafik data asli dengan hasil prediksi
        st.subheader('Grafik Data Asli dengan Hasil Prediksi')
        fig, ax = plt.subplots()
        data['#Passengers'].plot(style='--', color='gray', legend=True, label='Data Asli', ax=ax)
        forecast.plot(color='b', legend=True, label='Prediksi', ax=ax)
        st.pyplot(fig)
    with col2:
        # Tampilkan tabel hasil prediksi
        st.subheader('Tabel Hasil Prediksi')
        forecast_df = pd.DataFrame({
        'Tanggal': pd.date_range(start=data.index[-1], periods=forecast_steps, freq='M'),
        'Prediksi': forecast
        })
    st.dataframe(forecast_df)
