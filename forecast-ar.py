import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load the model
model = pickle.load(open('forecast-ar.sav', 'rb'))

# Load the historical data
df = pd.read_csv("AirPassengers.csv")
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df.set_index(['Month'], inplace=True)

st.title('Forecasting Penumpang Pesawat')
month = st.slider("Tentukan bulan", 1, 30, step=1)

pred = model.forecast(month)
pred = pd.DataFrame(pred, columns=['#Passengers'])

# Generate date range for the predictions
last_date = df.index[-1]
date_range = [last_date + timedelta(days=i*30) for i in range(1, month+1)]
pred['Month'] = date_range

if st.button("Predict"):
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(pred)
    with col2:
        fig, ax = plt.subplots()
        df['#Passengers'].plot(style='--', color='gray', legend=True, label='known')
        pred.plot(x='Month', y='#Passengers', color='b', legend=True, label='Prediction')
        st.pyplot(fig)
