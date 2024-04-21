import pandas as pd
import numpy as np


def create_rfm(dataframe, last_update_date):
    # Son gözlem tarihini hesapla
    dataframe['last_obs_date'] = pd.to_datetime(last_update_date) - pd.to_timedelta(dataframe['data_arc'], unit='D')
    max_date = dataframe['last_obs_date'].max()
    dataframe['recency'] = (max_date - dataframe['last_obs_date']).dt.days
    dataframe['recency_score'] = pd.qcut(dataframe['recency'], q=10, labels=False) + 1
    quantiles_n_obs_used = dataframe['n_obs_used'].quantile(q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    quantiles_H = dataframe['diameter'].quantile(q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # n_obs_used ve H için detaylı skorlama (1'den 10'a kadar)
    def n_obs_used_score(x):
        for i, threshold in enumerate(quantiles_n_obs_used, 1):
            if x <= threshold:
                return i
        return 10

    def H_score(x):
        for i, threshold in enumerate(quantiles_H[::-1], 1):
            if x > threshold:
                return i
        return 10  # Daha düşük 'H' değeri daha yüksek parlaklık anlamına gelir, bu yüzden ters çevrildi

    dataframe['recency_score'] = pd.qcut(dataframe['recency'], q=10, labels=False) + 1
    dataframe['frequency_score'] = dataframe['n_obs_used'].apply(n_obs_used_score)
    dataframe['monetary_score'] = dataframe['H'].apply(H_score)
    dataframe['rfm_score'] = dataframe['frequency_score'] + dataframe['monetary_score']+dataframe['recency_score']

    # Segmentasyon için daha geniş bir çeşitlilik sağlamak amacıyla yeni segment tanımları
    seg_map_detailed = {
        30: 'Çeşitli Fırsatlar',
        29: 'Sınırlı Değer, Yüksek Potansiyel',
        28: 'Değerli Fakat İzlenmekte',
        27: 'Yüksek Potansiyel, Riskli',
        26: 'Potansiyel, İzleme Gereken',
        25: 'Potansiyel Fakat Riskli',
        24: 'Değerli Fakat İzlenmekte',
        23: 'Potansiyel, İzleme Gereken',
        22: 'Yüksek Potansiyel, Riskli',
        21: 'Potansiyel Ancak İzlenmeli',
        20: 'Yüksek Potansiyel',
        19: 'Potansiyel Artırılabilir',
        18: 'Sınırlı Potansiyel',
        17: 'Çeşitlilik ve Potansiyel',
        16: 'Değerli Ancak İzlenmekte',
        15: 'Yüksek Potansiyel, Riskli',
        14: 'İzlem Gerektiren Potansiyel',
        13: 'Sınırlı Fakat İzlenmekte',
        12: 'Potansiyel, İzleme Gereken',
        11: 'Değerli Fakat Riskli',
        10: 'Potansiyel, İzleme Gereken',
        9: 'Değerli Fakat İzlenmekte',
        8: 'Değerli Fakat İzlenmekte',
        7: 'Potansiyel, İzleme Gereken',
        6: 'Potansiyel Fakat Riskli',
        5: 'Çeşitlilik ve Potansiyel',
        4: 'Sınırlı Değer, Yüksek Potansiyel',
        3: 'Değerli Fakat Riskli',
        2: 'Potansiyel, İzleme Gereken',
        1: 'Yüksek Potansiyel, Riskli'
    }

    dataframe['Broad_Segment'] = dataframe['rfm_score'].apply(lambda x: seg_map_detailed.get(x, 'Unknown'))

    # Segmentlere göre gözlem sayısını analiz edelim
    segment_analysis = dataframe['Broad_Segment'].value_counts().reset_index()
    segment_analysis.columns = ['Segment', 'Count']

    return segment_analysis, dataframe[
        ['name', 'n_obs_used', 'H', 'recency_score', 'frequency_score', 'monetary_score', 'rfm_score', 'Broad_Segment']]

##########################

"""   # RFM skorunu hesaplayarak birleştirme
    dataframe['rfm_score'] = dataframe['recency_score'] + dataframe['frequency_score'] + dataframe['monetary_score']

    return dataframe[['name', 'recency_score', 'frequency_score', 'monetary_score', 'rfm_score']]

    # Daha detaylı segment tanımları için fonksiyon


def segment_data_logarithmic(dataframe):
    seg_map_log = {
        3: 'Very Low Log RFM Score',
        4: 'Low Log RFM Score',
        5: 'Moderate Log RFM Score',
        6: 'High Log RFM Score',
        7: 'Very High Log RFM Score',
        8: 'Exceptionally High Log RFM Score',
        9: 'Extremely High Log RFM Score',
        10: 'Ultimate Log RFM Score'
    }

    dataframe['RFM_Segment_Log'] = dataframe['rfm_score'].apply(lambda x: seg_map_log.get(x, 'Unknown'))

    segment_analysis = dataframe['RFM_Segment_Log'].value_counts().reset_index()
    segment_analysis.columns = ['Segment', 'Count']

    return segment_analysis, dataframe[
        ['name', 'recency_score', 'frequency_score', 'monetary_score', 'rfm_score', 'RFM_Segment_Log']]


# Daha detaylı segment tanımları için fonksiyon
def segment_data_detailed(dataframe):
    seg_map_detailed = {
        3: 'Very Low RFM Score',
        4: 'Low RFM Score',
        5: 'Moderate RFM Score',
        6: 'High RFM Score',
        7: 'Very High RFM Score',
        8: 'Exceptionally High RFM Score',
        9: 'Extremely High RFM Score',
        10: 'Ultimate RFM Score'
    }

    dataframe['RFM_Segment'] = dataframe['rfm_score'].apply(lambda x: seg_map_detailed.get(x, 'Unknown'))

    segment_analysis = dataframe['RFM_Segment'].value_counts().reset_index()
    segment_analysis.columns = ['Segment', 'Count']

    return segment_analysis, dataframe[['name', 'recency_score', 'frequency_score', 'monetary_score', 'rfm_score', 'RFM_Segment']]



def asteroid_clv_score(dataframe, last_update_date):
    # RFM skorlarını hesapla
    dataframe = calculate_rfm_score(dataframe, last_update_date)

    # Monetary hesaplama
    #dataframe['monetary'] = dataframe['diameter']

    # CLV skoru hesaplama: RFM skorlarının ve Monetary değerlerinin ağırlıklı ortalaması alınarak bir CLV skoru oluşturulur.
    dataframe['clv_score'] = (dataframe['recency_score'] * 0.4) + (dataframe['frequency_score'] * 0.3) + (dataframe['monetary_score'] * 0.3)

    return dataframe[['name', 'recency_score', 'frequency_score', 'monetary_score', 'rfm_score', 'clv_score']]
    
"""