# Laporan Proyek Machine Learning
### Nama : Liza Puspita Putri
### Nim : 201351065
### Kelas : IF Malam B

## Domain Proyek

Pesawat yang menjadi transportasi udara yang lebih cepat dan efektif untuk digunakan saat perjalanan jauh terlebih ketika kita yang terburu-buru untuk segera sampai ke tempat tujuan, pesawat menjadi solusinya dibandingkan dengan menggunakan jalur darat.

## Business Understanding

Prediksi jumlah penumpang pesawat dari waktu kewaktu sangat amat dibutuhkan guna mengetahui demand yang akan terjadi. Dengan ini pembuatan aplikasi yang dapat memprediksi jumlah penumpang pesawat berdasarkan index waktu diharapkan dapat membantu siapapun yang membutuhkannya agar mengetahui kapan penumpang pesawat akan bertambah pesat.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Penumpang pesawat yang terus bertambah
- Penumpang pesawat yang bertambah saat beberapa waktu saja

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Dapat membantu siapapun yang membutuhkan informasi passengers demand berdasarkan index waktu
- Mempermudah mereka yang ingin merencanakan liburan dan ingin menggunakan pesawat untuk transport nya kapan passengers demand akan terjadi

    ### Solution statements
    - Pembuatan sistem yang mempermudah dalam memberikan informasi kapan pengumpang pesawat akan bertambah berdasarkan index waktu perbulannya sehingga siapapun yang membutuhkan informasi tersebut.
    - Sistem yang dibuat menggunakan dataset yang diambil dari kaggle dan diproses menggunakan 3 algoritma yang berbeda yang mana selanjutnya akan dipilih algoritma terbaik untuk dipakai didalam aplikasi tersebut

## Data Understanding
Dataset yang diambil dari kaggle ini berisi 2 kolom yaitu tanggal dan jumlah penumpang pesawat. 

Dataset: [Air Passengers](https://www.kaggle.com/datasets/rakannimer/air-passengers).

Dalam proses data understanding ini tahapan pertama yang dilakukan adalah:
1. import dataset

dikarenakan dataset diambil dari kaggle maka kita perlu import token/API kaggle kita:
```
from google.colab import files
files.upload()
```
lalu kita buat directory nya:
```
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
selanjutnya download datasetnya:
```
!kaggle datasets download -d rakannimer/air-passengers
```
setelah itu kita unzip file yang sudah di download:
```
!mkdir air-passengers
!unzip air-passengers.zip -d air-passengers
!ls air-passengers
```
jangan lupa untuk import library yang akan digunakan:
```
import pandas as pd
import numpy as np

# library untuk lvisualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# library untuk analisis time series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# library yang digunakan untuk forecasting
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
```
Setelah itu baru kita import datasetnya:
```
df = pd.read_csv("/content/air-passengers/AirPassengers.csv")
```
2. menampilkan 5 baris pertama dataset
```
df.head()
```
3. cek tipe data
```
df.info()
```
![Alt text](image.png)
4. cek ukuran dataset
```
df.shape
```
(144, 2)
dataset tersebut berisi 144 baris dengan 2 kolom
5. Null Check
```
df.isnull().sum()
```
Tidak terdapat data yang kosong

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- Month : Tanggal (Tahun-Bulan) ```object```
- #Passengers : Jumlah Penumpang Pesawat ```int64```

## Data Preparation
Dikarenakan kolom Month memiliki tipe data object maka harus kita convert menjadi datetime:
```
df['Month'] = pd.to_datetime(df['Month'])
```
lalu kita set index dari kolom month untuk menjadi acuan dalam melakukan forecasting:
```
df.set_index("Month",inplace=True)
```
kita tampilkan juga bagaimana grafik dari perubahan jumlah penumpang pasawatnya:
```
df['#Passengers'].plot(figsize=(12,5));
```
![Alt text](image-1.png)
Ternyata dalam beberapa periode waktu penumpang pesawat naik pesat dan juga grafik tersebut menunjukan jika penumpang pesawat cenderung naik.

Selanjutnya kita bagi terlebih dahulu antara train dan test data:
```
train = df.iloc[:100]
test = df.iloc[101:]
```

Setelah itu mari kita analisi dan memahami komponen-komponen utama dalam data deret waktu, yaitu tren, musiman, dan komponen residu (error)
```
decompose_add = seasonal_decompose(df['#Passengers'])
decompose_add.plot()
```
![Alt text](image-2.png)

Grafik ini berfungsi untuk mencerminkan perubahan jangka panjang atau kecenderungan dalam data. Dalam grafik ini dapat dilihat jika terdapat perubahan data yang cenderung naik.

Sekarang mari kita lihat selisih antara setiap dua poin data berturut-turut dalam sebuah rangkaian data deret waktu (time series):
```
diff_df = df.diff()
diff_df.head()
```
lalu kita hapus kolom yang berisi null values:
```
diff_df.dropna(inplace=True)
```
Sekarang kita lakukan uji adfuller yaitu uji statistik yang digunakan untuk mengevaluasi apakah sebuah deret waktu stasioner atau tidak:
```
result = adfuller(diff_df)
# The result is a tuple that contains various test statistics and p-values
# You can access specific values as follows:
adf_statistic = result[0]
p_value = result[1]

# Print the results
print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')
```
Selanjutnya kita cek korelasi dari deret waktunya:
```
plot_acf(diff_df)
plot_pacf(diff_df)
```
![Alt text](image-3.png)
![Alt text](image-4.png)

## Modeling
Ditahap modeling ini kita akan menggunakan 3 algoritma yang mana akan kita bandingkan algoritma terbaik yang selanjutnya akan dipakai untuk aplikasi tersebut.

kita akan coba untuk memprediksi 43 bulan kedepan:

  ### Single Exponential Smoothing
```
single_exp = SimpleExpSmoothing(train).fit()
single_exp_train_pred = single_exp.fittedvalues
single_exp_test_pred = single_exp.forecast(43)
```
```
train['#Passengers'].plot(style='--', color='gray', legend=True, label='train')
test['#Passengers'].plot(style='--', color='r', legend=True, label='test')
single_exp_test_pred.plot(color='b', legend=True, label='Prediction')
```
![Alt text](image-5.png)

```
Train_RMSE_SES = mean_squared_error(train, single_exp_train_pred)**0.5
Test_RMSE_SES = mean_squared_error(test, single_exp_test_pred)**0.5
Train_MAPE_SES = mean_absolute_percentage_error(train, single_exp_train_pred)
Test_MAPE_SES = mean_absolute_percentage_error(test, single_exp_test_pred)

print('Train RMSE :',Train_RMSE_SES)
print('Test RMSE :', Test_RMSE_SES)
print('Train MAPE :', Train_MAPE_SES)
print('Test MAPE :', Test_MAPE_SES)
```
Train RMSE : 23.47083303956671
Test RMSE : 106.96706722437959
Train MAPE : 0.08532342002218128
Test MAPE : 0.17254543771244724

  ## Double Exponential Smoothing
```
double_exp = ExponentialSmoothing(train, trend=None, initialization_method='heuristic', seasonal='add', seasonal_periods=29, damped_trend=False).fit()
double_exp_train_pred = double_exp.fittedvalues
double_exp_test_pred = double_exp.forecast(43)
```
```
train['#Passengers'].plot(style='--', color='gray', legend=True, label='train')
test['#Passengers'].plot(style='--', color='r', legend=True, label='test')
double_exp_test_pred.plot(color='b', legend=True, label='Prediction')
```
![Alt text](image-6.png)

```
Train_RMSE_DES = mean_squared_error(train, double_exp_train_pred)**0.5
Test_RMSE_DES = mean_squared_error(test, double_exp_test_pred)**0.5
Train_MAPE_DES = mean_absolute_percentage_error(train, double_exp_train_pred)
Test_MAPE_DES = mean_absolute_percentage_error(test, double_exp_test_pred)

print('Train RMSE :',Train_RMSE_DES)
print('Test RMSE :', Test_RMSE_DES)
print('Train MAPE :', Train_MAPE_DES)
print('Test MAPE :', Test_MAPE_DES)
```
Train RMSE : 23.283893193337274
Test RMSE : 94.57214255933388
Train MAPE : 0.07900374086543273
Test MAPE : 0.15438871066201712

  ## ARIMA
```
ar = ARIMA(train, order=(15,1,15)).fit()
ar_train_pred = ar.fittedvalues
ar_test_pred = ar.forecast(43)
```
```
train['#Passengers'].plot(style='--', color='gray', legend=True, label='train')
test['#Passengers'].plot(style='--', color='r', legend=True, label='test')
ar_test_pred.plot(color='b', legend=True, label='Prediction')
```
![Alt text](image-7.png)
```
Train_RMSE_AR = mean_squared_error(train, ar_train_pred)**0.5
Test_RMSE_AR = mean_squared_error(test, ar_test_pred)**0.5
Train_MAPE_AR = mean_absolute_percentage_error(train, ar_train_pred)
Test_MAPE_AR = mean_absolute_percentage_error(test, ar_test_pred)

print('Train RMSE :',Train_RMSE_AR)
print('Test RMSE :', Test_RMSE_AR)
print('Train MAPE :', Train_MAPE_AR)
print('Test MAPE :', Test_MAPE_AR)
```
Train RMSE : 14.20071832771583
Test RMSE : 45.285402548094446
Train MAPE : 0.04423659596567478
Test MAPE : 0.0929043309516595

Selanjutnya mari kita evaluasi 3 algoritma tersebut

## Evaluation
Pada tahap evaluasi ini kita akan membandingkan nilai Root Mean Squared Error (RMSE) dan Mean Absolute Percentage Error (MAPE) yang mana selanjutnya akan kita urutkan mana nilai RMSE yang paling kecil maka algoritma tersebut yang akan kita pakai:
```
comparision_df = pd.DataFrame(data=[
    ['Single Exp Smoothing', Test_RMSE_SES, Test_MAPE_SES],
    ['Double Exp Smoothing', Test_RMSE_DES, Test_MAPE_DES],
    ['ARIMA', Test_RMSE_AR, Test_MAPE_AR]
    ],
    columns=['Model', 'RMSE', 'MAPE'])
comparision_df.set_index('Model', inplace=True)
```
```
comparision_df.sort_values(by='RMSE')
```
![Alt text](image-8.png)

dapat dilihat jika nilai RMSE dan MAPE ada pada algoritma ARIMA, maka dari itu algoritma yang akan dipakai adalah algoritma ARIMA

## Deployment
Link Aplikasi: [ARIMA Forecasting App](https://forecast-ar.streamlit.app/)

