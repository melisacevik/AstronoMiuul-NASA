import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

def predict_diameter(csv_file_path):
    # Veri setini yükleme
    df = pd.read_csv(csv_file_path, low_memory=False)

    # Sayısal olması gereken sütunlarda string değerleri temizleme
    numeric_cols = ['a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 'condition_code',
                    'n_obs_used', 'H', 'albedo', 'rot_per', 'GM', 'moid', 'n', 'per', 'ma']  # 'diameter' hariç

    # Sayısal sütunlardaki non-numeric değerleri NaN ile değiştirme
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Kategorik değişkenler için label encoding
    categorical_features = ['neo', 'pha', 'class', 'spec_B', 'spec_T']
    df[categorical_features] = df[categorical_features].apply(lambda x: x.astype('category').cat.codes)

    # Eksik değerleri median ile doldurma
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # 'diameter' sütunu için tahmin yapılacak veri kontrolü
    train_data = df[df['diameter'].notna()]
    predict_data = df[df['diameter'].isna()]

    if predict_data.empty:
        st.write("No missing diameter values to predict.")
        return

    X_train = train_data.drop(columns=['diameter', 'name', 'extent', 'GM'])
    y_train = train_data['diameter']
    X_predict = predict_data.drop(columns=['diameter', 'name', 'extent', 'GM'])

    # Veri setini eğitim ve doğrulama setlerine bölme
    X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(X_train, y_train, test_size=0.2,
                                                                          random_state=42)

    # Veri ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_part)
    X_val_scaled = scaler.transform(X_val_part)
    X_predict_scaled = scaler.transform(X_predict)

    # Model eğitimi
    model = LGBMRegressor(random_state=42)
    model.fit(X_train_scaled, y_train_part)

    # Doğrulama seti üzerinde tahmin
    y_pred_val = model.predict(X_val_scaled)

    # Performans metriklerini hesaplama
    mse = mean_squared_error(y_val_part, y_pred_val)
    r2 = r2_score(y_val_part, y_pred_val)
    st.write(f"Validation MSE: {mse:.2f}")
    st.write(f"Validation R²: {r2:.2f}")

    # Eksik 'diameter' değerlerinin tahmini
    predicted_diameters = model.predict(X_predict_scaled)

    # Tahmin edilen değerlerle 'diameter' sütununun doldurulması
    df.loc[df['diameter'].isna(), 'diameter'] = predicted_diameters

    # Güncellenmiş veri setini yeni bir CSV dosyası olarak kaydetme
    completed_csv_file_path = csv_file_path.replace(".csv", "_completed.csv")
    df.to_csv(completed_csv_file_path, index=False)

    st.write("Boş çap (diameter) verileri dolduruldu ve dosya oluşturuldu.")
    return completed_csv_file_path


