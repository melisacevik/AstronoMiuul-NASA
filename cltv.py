import streamlit as st
import pandas as pd


def calculate_cltv(df):
    # 'name' değişkeni boş olmayanlar için filtreleme yap
    df_filtered = df[df['name'].notnull()]

    # Diameter değerine göre sıralama yap
    df_sorted_diameter = df_filtered.sort_values(by='diameter', ascending=False)

    # Recency (data_arc) değerine göre sıralama yap
    df_sorted_recency = df_filtered.sort_values(by='data_arc', ascending=False)

    # Frequency (n_obs_used) değerine göre sıralama yap
    df_sorted_frequency = df_filtered.sort_values(by='n_obs_used', ascending=False)

    # Her bir asteroid için CLTV skorunu hesapla
    cltv_scores = (df_sorted_diameter.index + df_sorted_recency.index + df_sorted_frequency.index) / 3

    # Oluşturulan CLTV skorlarını veri setine ekle
    df_filtered['cltv_score'] = cltv_scores

    # CLTV skoruna göre veri setini sırala
    df_sorted_cltv = df_filtered.sort_values(by='cltv_score', ascending=False)

    return df_sorted_cltv