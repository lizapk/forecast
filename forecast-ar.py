# Import library yang diperlukan
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Fungsi untuk memuat data dan model ARIMA
def load_data_and_model(data_file, model_file):
    # Muat data time series
    data = pd.read_csv("AirPassengers.csv")
    
    # Muat model ARIMA
    model = ARIMA
    model = model.load("forecast-ar.sav")
    
    return data, model

# Fungsi untuk menampilkan grafik
def plot_data_and_forecast(data, model, forecast_steps):
    # Dekomposisi musiman
    result = seasonal_decompose(data, model='additive')
    
    # Plot data asli
    st.subheader('Data Asli')
    st.line_chart(data)
    
    # Plot hasil dekomposisi
    st.subheader('Hasil Dekomposisi')
    st.line_chart(result.trend)
    st.line_chart(result.seasonal)
    st.line_chart(result.resid)
    
    # Prediksi
    forecast, stderr, conf_int = model.forecast(steps=forecast_steps)
    
    # Plot hasil prediksi
    st.subheader('Hasil Prediksi')
    st.line_chart(forecast)
    
    # Tampilkan tabel hasil prediksi
    st.subheader('Tabel Hasil Prediksi')
    forecast_df = pd.DataFrame({
        'Waktu': pd.date_range(start=data.index[-1], periods=forecast_steps + 1, closed='right'),
        'Prediksi': forecast
    })
    st.dataframe(forecast_df)

# Main Streamlit app
def main():
    st.title('ARIMA Forecasting App')
    
    data_file = st.file_uploader('Upload file data time series (CSV)', type=['csv'])
    model_file = st.file_uploader('Upload file model ARIMA', type=['pkl'])
    
    if data_file is not None and model_file is not None:
        data, model = load_data_and_model(data_file, model_file)
        
        forecast_steps = st.slider('Jumlah Langkah Prediksi', 1, 365, 30)
        
        plot_data_and_forecast(data, model, forecast_steps)

if __name__ == '__main__':
    main()
