import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('forecast-ar.sav','rb'))

df = pd.read_csv("AirPassengers.csv")
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df.set_index(['Month'], inplace=True)

st.title('Forecasting Penumpang Pesawat')
month = st.slider("Tentukan bulan",1,12, step=1)

pred = model.forecast(month)
pred = pd.DataFrame(pred, columns=['#Passengers'])

if st.button("Predict"):

    col1, col2 = st.columns([2, 3])

    # Convert the 'Month' column to a datetime format in the pred DataFrame
    pred['Month'] = pd.date_range(start=df.index[-1], periods=month+1, freq='M')
    pred.set_index(['Month'], inplace=True)

    # Ensure 'Passengers' column is numeric
    pred['#Passengers'] = pred['#Passengers'].astype(float)

    with col1:
        st.dataframe(pred)
    with col2:
        fig, ax = plt.subplots()
        df['#Passengers'].plot(style='--', color='gray', legend=True, label='known')
        pred['#Passengers'].plot(color='b', legend=True, label='Prediction')
        st.pyplot(fig)