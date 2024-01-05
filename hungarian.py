#calls local : python -m streamlit run hungarian.py
import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle
#Load Data Hungarian.data
with open("hungarian.data", encoding='Latin1') as file:
  lines = [line.strip() for line in file]
#Melihat data frame
data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

df = pd.DataFrame.from_records(data)
#Penghapusan nilai null
df = df.iloc[:, :-1]
df = df.drop(df.columns[0], axis=1)
#Merubah Nilai menjadi float
df = df.astype(float)
#Menganti nilai -9.0
df.replace(-9.0, np.NaN, inplace=True)
#Memilih attribute yang digunakan
df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]
#Memberikan nama tiap kolom
column_mapping = {
  2: 'age',
  3: 'sex',
  8: 'cp',
  9: 'trestbps',
  11: 'chol',
  15: 'fbs',
  18: 'restecg',
  31: 'thalach',
  37: 'exang',
  39: 'oldpeak',
  40: 'slope',
  43: 'ca',
  50: 'thal',
  57: 'target'
}

df_selected.rename(columns=column_mapping, inplace=True)
#Mengapus kolom ca,slope,dan thal
columns_to_drop = ['ca', 'slope','thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)
#Menghitung nilai rata"
meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()
#Merubah tipe data menjadi float
meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)
meanRestCG = meanRestCG.astype(float)
#Membulatkan nilai rata"
meanTBPS = round(meanTBPS.mean())
meanChol = round(meanChol.mean())
meanfbs = round(meanfbs.mean())
meanthalach = round(meanthalach.mean())
meanexang = round(meanexang.mean())
meanRestCG = round(meanRestCG.mean())
#Mengisi nilai value dengan rata" yang sebelumnya telah dihitung
fill_values = {
  'trestbps': meanTBPS,
  'chol': meanChol,
  'fbs': meanfbs,
  'thalach':meanthalach,
  'exang':meanexang,
  'restecg':meanRestCG
}

df_clean = df_selected.fillna(value=fill_values)
#Membersihkan data yang terduplikasi
df_clean.drop_duplicates(inplace=True)
#Menentukan variabel input dan target
X = df_clean.drop("target", axis=1)
y = df_clean['target']
#Membuat oversampling dengan smote
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
#Memanggil model xgboost untuk hasil prediksi
model = pickle.load(open("xgb_model.pkl", 'rb'))
#Menentukan nilai akurasi
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

df_final = X
df_final['target'] = y

# ========================================================================================================================================================================================

# STREAMLIT
#Menganti title dan icon pada web streamlit
st.set_page_config(
  page_title = "Hungarian-Heart-Disease",
  page_icon = ":skull:"
)
#Memberikan tittle
st.title(":heart: :red[Hungarian Heart Disease] :heart:")
if accuracy > 80:
  st.write(f"Model Akurasi :  :green[**{accuracy}**]% :green[High]:smile:")
elif accuracy < 79:
  st.write(f"Model Akurasi :  :red[**{accuracy}**]% :red[Low]:sad:")
st.write("Jenis Model : :blue[**XGBOOST**]")
st.write("")
#Membuat 2 tab
tab1, tab2 = st.tabs(["[Single]", "[Multi]"])

with tab1:
  #Membuat sidebar untuk menginput data
  st.sidebar.header("Input Data")
  #Memasukan input nilai umur
  age = st.sidebar.number_input(label=":blue[**Age**]", min_value=df_final['age'].min(), max_value=df_final['age'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['age'].min()}**], :red[Max] value: :red[**{df_final['age'].max()}**]")
  st.sidebar.write("")
  #Memasukan jenis kelamin
  sex_sb = st.sidebar.selectbox(label=":blue[**Sex**]", options=["Male", "Female"])
  st.sidebar.write("")
  st.sidebar.write("")
  if sex_sb == "Male":
    sex = 1
  elif sex_sb == "Female":
    sex = 0
  # -- Value 0: Female
  # -- Value 1: Male
  #Memilih jenis nyeri pada dada
  cp_sb = st.sidebar.selectbox(label=":blue[**Chest pain type**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
  st.sidebar.write("")
  st.sidebar.write("")
  if cp_sb == "Typical angina":
    cp = 1
  elif cp_sb == "Atypical angina":
    cp = 2
  elif cp_sb == "Non-anginal pain":
    cp = 3
  elif cp_sb == "Asymptomatic":
    cp = 4
  # -- Value 1: typical angina
  # -- Value 2: atypical angina
  # -- Value 3: non-anginal pain
  # -- Value 4: asymptomatic
  #Input nilai tensi darah
  trestbps = st.sidebar.number_input(label=":blue[**Resting blood pressure** (in mm Hg on admission to the hospital)]", min_value=df_final['trestbps'].min(), max_value=df_final['trestbps'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['trestbps'].min()}**], :red[Max] value: :red[**{df_final['trestbps'].max()}**]")
  st.sidebar.write("")
  #Input Nilai Kolesterol
  chol = st.sidebar.number_input(label=":blue[**Serum cholestoral** (in mg/dl)]", min_value=df_final['chol'].min(), max_value=df_final['chol'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['chol'].min()}**], :red[Max] value: :red[**{df_final['chol'].max()}**]")
  st.sidebar.write("")
  #Input nilai tes gula
  fbs_sb = st.sidebar.selectbox(label=":blue[**Fasting blood sugar > 120 mg/dl?**]", options=["False", "True"])
  st.sidebar.write("")
  st.sidebar.write("")
  if fbs_sb == "False":
    fbs = 0
  elif fbs_sb == "True":
    fbs = 1
  # -- Value 0: false
  # -- Value 1: true
  #Input apakah ada kelainan pada jantung
  restecg_sb = st.sidebar.selectbox(label=":blue[**Resting electrocardiographic results**]", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
  st.sidebar.write("")
  st.sidebar.write("")
  if restecg_sb == "Normal":
    restecg = 0
  elif restecg_sb == "Having ST-T wave abnormality":
    restecg = 1
  elif restecg_sb == "Showing left ventricular hypertrophy":
    restecg = 2
  # -- Value 0: normal
  # -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV)
  # -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
  #Input nilai detang jantung tertinggi
  thalach = st.sidebar.number_input(label=":blue[**Maximum heart rate achieved**]", min_value=df_final['thalach'].min(), max_value=df_final['thalach'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['thalach'].min()}**], :red[Max] value: :red[**{df_final['thalach'].max()}**]")
  st.sidebar.write("")
  #Input apakah ada penyakit induced angina?
  exang_sb = st.sidebar.selectbox(label=":blue[**Exercise induced angina?**]", options=["No", "Yes"])
  st.sidebar.write("")
  st.sidebar.write("")
  if exang_sb == "No":
    exang = 0
  elif exang_sb == "Yes":
    exang = 1
  # -- Value 0: No
  # -- Value 1: Yes
  #Input nilai depresi ST yang diakibatkan oleh latihan relative terhadap saat istirahat
  oldpeak = st.sidebar.number_input(label=":blue[**ST depression induced by exercise relative to rest**]", min_value=df_final['oldpeak'].min(), max_value=df_final['oldpeak'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['oldpeak'].min()}**], :red[Max] value: :red[**{df_final['oldpeak'].max()}**]")
  st.sidebar.write("")
  #Menyimpan nilai ke dataframe dari hasil input
  data = {
    'Age': age,
    'Sex': sex_sb,
    'Chest pain type': cp_sb,
    'RPB': f"{trestbps} mm Hg",
    'Serum Cholestoral': f"{chol} mg/dl",
    'FBS > 120 mg/dl?': fbs_sb,
    'Resting ECG': restecg_sb,
    'Maximum heart rate': thalach,
    'Exercise induced angina?': exang_sb,
    'ST depression': oldpeak,
  }
  #Menampilkan hasil input sebelumnya ke dataframe
  preview_df = pd.DataFrame(data, index=['input'])
  
  st.header("Detail Input Pengguna")
  st.write("")
  st.dataframe(preview_df.iloc[:, :6])
  st.write("")
  st.dataframe(preview_df.iloc[:, 6:])
  st.write("")

  result = ":violet[-]"
  #Membuat button prediksi
  predict_btn = st.button("**Predict**", type="primary")

  st.write("")
  if predict_btn:
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
    prediction = model.predict(inputs)[0]
    
    bar = st.progress(0)
    status_text = st.empty()
    #Membuat proses loading prediksi
    for i in range(1, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()
    #Menampilkan hasil prediksi dari 0-4 yang menunjukan tingkat penyakit jantung
    if prediction == 0:
      result = ":green[**Healthy**]"
    elif prediction == 1:
      result = ":orange[**Heart disease level 1**]"
    elif prediction == 2:
      result = ":orange[**Heart disease level 2**]"
    elif prediction == 3:
      result = ":red[**Heart disease level 3**]"
    elif prediction == 4:
      result = ":red[**Heart disease level 4**]"
  #Menampilkan hasil prediksi
  st.write("")
  st.write("")
  st.subheader("Prediction:")
  st.subheader(result)
#Multi Prediksi
with tab2:
  st.header("Predict multiple data:")
  #Membuat sample data pada csv
  sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')
  #Membuat button download csv
  st.write("")
  st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')
  #Membuat upload csv untuk multi prediksi
  st.write("")
  st.write("")
  file_uploaded = st.file_uploader("Upload a CSV file", type='csv')
  #Jika file csv sudah dibaca maka akan dibuat hasil prediksinya
  if file_uploaded:
    uploaded_df = pd.read_csv(file_uploaded)
    prediction_arr = model.predict(uploaded_df)

    bar = st.progress(0)
    status_text = st.empty()
    #Membuat proses loading saat upload
    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    result_arr = []
    #Menampilkan hasil prediksi dari 0-4 yang menunjukan tingkat penyakit jantung
    for prediction in prediction_arr:
      if prediction == 0:
        result = "Healthy"
      elif prediction == 1:
        result = "Heart disease level 1"
      elif prediction == 2:
        result = "Heart disease level 2"
      elif prediction == 3:
        result = "Heart disease level 3"
      elif prediction == 4:
        result = "Heart disease level 4"
      result_arr.append(result)
    #Hasil multi prediksi akan dibuat menjadi dataframe
    uploaded_result = pd.DataFrame({'Prediction Result': result_arr})
    #Membuat proses loading saat akan memprediksi
    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()
    
    col1, col2 = st.columns([1, 2])
    #pada column 1 nantinya akan menampilkan hasil resultnya
    #pada column 2 akan menampilkan input user yang berbentuk dataframe
    with col1:
      st.dataframe(uploaded_result)
    with col2:
      st.dataframe(uploaded_df)
