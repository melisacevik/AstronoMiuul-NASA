import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from difrfm import create_rfm
from datasetOrbit import datasetOrbit
from datetime import time
import base64
import random
from sklearn.metrics import  mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import craterPredict
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import string
from cltv import calculate_cltv
from lgbm_askom2 import predict_diameter
from asteroid_simulation import CustomPage, asteroid_orbit_simulation
from sklearn.impute import SimpleImputer
import time
from predictVelocity import calculate_orbital_velocity
import streamlit as st
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#InsterstellarProject/asteroid/asteroid.py

# Set page config
#st.set_page_config(layout="wide", page_title="Asteroid", page_icon="🌠")


@st.cache_data
def get_data(processed, sample_size=None):
    # Veriyi yükle
    data = pd.read_csv('InsterstellarProject/asteroid/asteroid.csv')
    data_completed = pd.read_csv('InsterstellarProject/asteroid/asteroid_completed.csv')

    if processed == 0:
        data['name'] = data['name'].apply(lambda x: ''.join(random.choices(string.ascii_letters, k=5)) if pd.isnull(x) else x)
        # Gerekli sütunlarda eksik değerleri olan satırları silme
        data = data.dropna(subset=['a', 'q', 'e', 'i', 'om', 'w'])
        # sample_size'a göre rastgele örnek seçme

    elif processed == 1:
        data['name'] = data['name'].apply(lambda x: ''.join(random.choices(string.ascii_letters, k=5)) if pd.isnull(x) else x)

    elif processed == 2:
        data['diameter'] = pd.to_numeric(data['diameter'],
                                         errors='coerce')  # diameter sütununu sayısal bir türe dönüştürme
        data = data.dropna(subset=['diameter'])  # NaN değerlere sahip satırları kaldırma

    elif processed == 3:
        data['diameter'] = pd.to_numeric(data['diameter'],
                                         errors='coerce')  # diameter sütununu sayısal bir türe dönüştürme

    elif processed == 4:
        data_completed['diameter'] = pd.to_numeric(data_completed['diameter'],
                                         errors='coerce')  # diameter sütununu sayısal bir türe dönüştürme




        return data_completed

    elif processed == 5:
        data_completed['name'] = data_completed['name'].apply(lambda x: ''.join(random.choices(string.ascii_letters, k=5)) if pd.isnull(x) else x)
        # Gerekli sütunlarda eksik değerleri olan satırları silme
        data_completed = data_completed.dropna(subset=['a', 'q', 'e', 'i', 'om', 'w','diameter'])
        # sample_size'a göre rastgele örnek seçme
        return data_completed


    elif processed == 6:
        data['name'] = data['name'].apply(
            lambda x: ''.join(random.choices(string.ascii_letters, k=5)) if pd.isnull(x) else x)
        # Gerekli sütunlarda eksik değerleri olan satırları silme
        data = data.dropna(subset=['a', 'q', 'e', 'i', 'om', 'w', 'diameter'])
        numeric_cols = ['a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 'condition_code',
                        'n_obs_used', 'H', 'albedo', 'rot_per', 'GM', 'moid', 'n', 'per', 'ma', 'diameter']

        # Sayısal sütunlardaki non-numeric değerleri NaN ile değiştirme
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data['data_arc'] = pd.to_numeric(data['data_arc'],
                                                   errors='coerce')  # diameter sütununu sayısal bir türe dönüştürme
        data['diameter'] = pd.to_numeric(data['diameter'],
                                                   errors='coerce')  # diameter sütununu sayısal bir türe dönüştürme
        return data



    elif sample_size is not None:
        data = data.sample(sample_size)

    return data





def generate_stars_css(num_stars=300):
    stars_css = ""
    for _ in range(num_stars):
        x = random.randint(0, 100)  # Viewport width percentage
        y = random.randint(0, 100)  # Viewport height percentage
        stars_css += f"{x}vw {y}vh #FFF, "
    stars_css = stars_css.rstrip(", ")
    return f".star {{ width: 1px; height: 1px; background: transparent; box-shadow: {stars_css}; position: fixed; }}"


def calculate_rfm_scores(df):
    # Recency skorunu hesapla
    df['Recency_Score'] = pd.qcut(df['per'], q=4, labels=False)

    # Frequency skorunu hesapla
    df['Frequency_Score'] = pd.qcut(df['n_obs_used'], q=4, labels=False, duplicates='drop')

    # Monetary skorunu hesapla
    df['Monetary_Score'] = pd.qcut(df['H'], q=4, labels=False)

    # RFM skorunu hesapla
    df['RFM_Score'] = df['Recency_Score'] + df['Frequency_Score'] + df['Monetary_Score']

    return df






def generate_meteors_css_and_animation(num_meteors=15):
    meteors_css = ""
    for i in range(1, num_meteors + 1):
        h = random.randint(50, 250)  # top position
        v = random.randint(9, 99)  # left position
        d = random.uniform(3.0, 7.3)  # duration
        meteors_css += f"""
        .meteor-{i} {{
            position: fixed;
            top: {h}px;
            left: {v}%;
            width: 1px;
            height: 1px;
            background-image: linear-gradient(to right, #fff, rgba(255,255,255,0));
            transform: rotate(-45deg);
            animation: meteor-{i} {d}s linear infinite;
        }}
        .meteor-{i}:before {{
            content: "";
            position: absolute;
            width: 4px;
            height: 5px;
            border-radius: 50%;
            margin-top: -2px;
            margin-left: -2px;
            background: rgba(255, 255, 255, .7);
            box-shadow: 0 0 15px 3px #fff;
        }}
        @keyframes meteor-{i} {{
            0% {{
                opacity: 1;
                transform: translate3d(-300px, -300px, 0) rotate(-45deg);
            }}
            100% {{
                opacity: 0;
                transform: translate3d(100vw, 100vh, 0) rotate(-45deg);
            }}
        }}
        """
    return meteors_css

# CSS to style the body and common styles for stars and meteors
base_css = """
html, body {
    height: 100%;
    overflow: hidden;
    background-image: radial-gradient(ellipse at top , #080e21 0%,  #1b2735 95%);
}
.star, .meteor {
    position: absolute;
}
"""

# Generating CSS
stars_css = generate_stars_css()
meteors_css = generate_meteors_css_and_animation()


# Combine all CSS
final_css = base_css + stars_css + meteors_css

# HTML for stars and meteors
html_content = '<div class="star"></div>' + ''.join([f'<div class="meteor meteor-{i}"></div>' for i in range(1, 16)])

# Use in Streamlit
st.markdown(f"<style>{final_css}</style>", unsafe_allow_html=True)
st.markdown(html_content, unsafe_allow_html=True)

center_css = """
<style>
.center {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
}
html, body {
    height: 100%;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
"""

st.markdown("""
    <style>
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px; /* İstediğiniz yüksekliği ayarlayabilirsiniz */
        }
        h1 {
            text-align: center;
        }
        .rainbow-divider {
            background: linear-gradient(90deg, #ff0000, #ff7700, #ffdd00, #33ff00, #0099ff, #7700ff);
            height: 5px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)



# Load data



# Define functions
def tabii_efendim(diameter_km, velocity_km_s=100, density_kg_m3=300000, k=0.01):
    """
    Göktaşı çarpması sonucu oluşacak enerjiyi, göktaşının kütlesini, yaratacağı kraterin yarıçapını ve yüzey alanını hesaplayan fonksiyon.
    """
    import numpy as np

    pi = np.pi
    kt_to_joule = 4.184e12  # 1 kiloton TNT'nin joule cinsinden enerjisi

    # Çapı metre cinsine çevirme
    d_m = diameter_km * 1000

    # Kütle hesaplama
    mass_kg = (4 / 3) * pi * (d_m / 2) ** 3 * density_kg_m3

    # Çarpma hızını m/s cinsine çevirme
    v_m_s = velocity_km_s * 1000

    # Çarpma enerjisi hesaplama (joule cinsinden)
    impact_energy_joule = 0.5 * mass_kg * v_m_s ** 2

    # Enerjiyi kiloton TNT eşdeğeri olarak çevirme
    impact_energy_kt = impact_energy_joule / kt_to_joule

    # Krater yarıçapını hesaplama
    crater_radius_m = k * (impact_energy_joule ** 0.25)

    # Kraterin yüzey alanını hesaplama
    crater_surface_area_m2 = pi * (crater_radius_m ** 2)

    return mass_kg, impact_energy_joule, impact_energy_kt, crater_radius_m, crater_surface_area_m2

def eksik_veri_detay(df):
    """
    Eksik veri detaylarını hesaplar.
    """
    eksik_veri = df.isnull().sum()
    toplam_veri = len(df)
    eksik_veri_orani = 100 * eksik_veri / toplam_veri
    eksik_veri_tablosu = pd.concat([eksik_veri, eksik_veri_orani, pd.Series(toplam_veri, index=df.columns)], axis=1)
    eksik_veri_tablosu_renamed = eksik_veri_tablosu.rename(columns = {0 : 'Eksik Değerler', 1 : 'Oran (%)', 2: 'Toplam Veri'})
    eksik_veri_tablosu_renamed = eksik_veri_tablosu_renamed[eksik_veri_tablosu_renamed.iloc[:,1] != 0].sort_values('Oran (%)', ascending=False)
    return eksik_veri_tablosu_renamed

def refresh_data():
    """
    Yeni bir örneklem alır ve görselleştirmeyi yeniler.
    """
    # Yeni bir örneklem seç
    sample_df = df.sample(n=10)
    sample_df.to_csv('sample.csv', index=False)

    # Görselleştirme ve animasyonu yeniden oluştur
    datasetOrbit.fileName("sample.csv")
    datasetOrbit.datasetCalculateOrbit(plot_steps=1000, n_orbits=12, color="yellow", random_color=True, trajectory=True, sun=True, delimiter=",")
    datasetOrbit.datasetAnimateOrbit(dpi=250, save=True, export_zoom=3, font_size="xx-small")

    # Animasyonu Streamlit'te göster
    file_path = "Asteroid-orbit.gif"
    with open(file_path, "rb") as file:
        contents = file.read()
        data_url = base64.b64encode(contents).decode("utf-8")
    centered_html = f"""
    <div style="display: flex; justify-content: center; align-items: center;">
        <img src="data:image/gif;base64,{data_url}" alt="Alt Text">
    </div>
    """
    st.markdown(centered_html, unsafe_allow_html=True)

def get_sidebar_image(page):
    if page == "Hosgeldiniz":
        return "InsterstellarProject/asteroid/Images/astronotlar.jpeg"
    elif page == "Veriseti Hikayesi":
        return "InsterstellarProject/asteroid/Images/asteroid.jpeg"
    elif page == "Simulation":
        return "InsterstellarProject/asteroid/Images/sidebar.jpeg"
    elif page == "Krater Alan":
        return "InsterstellarProject/asteroid/Images/krater-photo.jpeg"
    elif page == "Diameter Tahmini":
        return "InsterstellarProject/asteroid/Images/diameter-predict.jpeg"
    elif page == "Gunes ve Asteroidler":
        return "InsterstellarProject/asteroid/Images/sun-and-asteroid.jpeg"
    elif page == "RFM-ALTV":
        return "InsterstellarProject/asteroid/Images/CLTV.jpeg"



def cemberde_krater_ciz(dunya_cap_km, krater_yaricap_km):
    """
    Dünya üzerinde bir kraterin çizimini yapar.
    """
    fig, ax = plt.subplots()

    # Dünya'yı temsil eden çember
    dunya = Circle((0.5, 0.5), 0.4, color='blue', label='Dünya')
    ax.add_artist(dunya)

    # Krateri temsil eden çember
    # Kraterin yeri ve boyutu semboliktir.
    krater = Circle((0.5, 0.5), krater_yaricap_km / dunya_cap_km * 0.4, color='red', label='Krater')
    ax.add_artist(krater)

    # Grafiği düzenleme
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')  # Eksenleri gizleme
    ax.legend()  # Efsane eklemek için

    return fig

# Define layout
st.sidebar.title('AstronoMiuul')

selected_page = st.sidebar.radio("Menu", ["Hosgeldiniz", "Veriseti Hikayesi" , "Simulation", "Krater Alan", "Diameter Tahmini", "Gunes ve Asteroidler", "RFM-ALTV"])
sidebar_image = get_sidebar_image(selected_page)
st.sidebar.image(sidebar_image, use_column_width=True)


# Welcome Tab
if selected_page == "Hosgeldiniz":
    st.title('Hoşgeldiniz!')
    # Yolu görselin bulunduğu yere göre ayarla
    image_path = "InsterstellarProject/asteroid/Images/astronotlar.jpeg"

    # Sütunlar oluştur
    col1, col2, col3 = st.columns([1, 6, 1])

    # Orta sütuna görseli yerleştir
    with col2:
        st.image(image_path, use_column_width=True)
    st.write("")
    st.markdown("İş Problemi: ")
    st.write("Asteroidlerin Yörünge, Tehlike ve Potansiyel Maden Analizi")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("Çözüm Yaklaşımı:")
    st.write("Veri Analizi: Asteroid veri seti üzerinde keşifsel veri analizi yaparak yörünge detaylarını tespit etmek ve potansiyel madencilik adaylarını belirlemek.")
    st.write("Tahmin Modelleri: Makine öğrenimi modelleri kullanarak madencilik potansiyeli yüksek asteroidleri tahmin etmek.")






# Dataset Story Tab
elif selected_page == "Veriseti Hikayesi":
    st.title('NASA Veriseti Hikayesi')
    st.write("")
    st.write("NASA'ya bağlı California Institute of Technology Jet Propulsion Laboratory tarafından oluşturulan resmi veri seti kullanıldı. ")
    veri_seti_detay = get_data(processed=1)
    satir_sayisi, sutun_sayisi = veri_seti_detay.shape
    st.write("Gözlem Birimi: ", satir_sayisi)
    st.write("Değişken Sayısı: ", sutun_sayisi)
    st.write("")
    st.write("")
    st.table(veri_seti_detay.head())






# Asteroid Impact Scenarios Tab
elif selected_page == "Simulation":
    st.title('Asteroid Çarpma Senaryoları')
    st.write("""
    - **Küçük Asteroidler**: Atmosfere giriş sağladıklarında yanarak küçülür ve daha zararsız hale gelirler.
    - **Büyük Asteroids**: Küresel çapta felaketlere yol açabilirler.
    """)
    #burayı sil ve aşağıda argüman df_sim sil
    df_sim = get_data(processed=4)
    # Simülasyon başlatma butonu
    st.session_state['active_page'] = 'Simulation'
    if st.session_state['active_page'] == 'Simulation':
        simulation_page = CustomPage("Simulation")
        asteroid_orbit_simulation(simulation_page,df_sim)



    #df_car_sen = get_data(processed=2)
    #impact_scenarios = df_car_sen.sample(n=5)  # Örnek bir seçim
    #fig = px.scatter(impact_scenarios, x='diameter', y='moid', size='diameter', color='condition_code', title='Simulation')
    #st.plotly_chart(fig)



# Possible Crater Tab
elif selected_page == "Krater Alan":
    st.title('Krater Alan Ölçü Tahmini')
    st.write("""
        Bu bölümde bir göktaşının çarpması sonucu oluşacak kraterin yüzey alanı için tahmin oluşturabilirsiniz.
    """)


    # Hız sütunu ekleme

    df_pred_vel = get_data(processed=4)
    df_pred_vel['diameter'] = df_pred_vel['diameter'].fillna(0).apply(round).astype(int)
    df_pred_vel['pred_velocity'] = df_pred_vel['a'].apply(calculate_orbital_velocity)


    df_muh_kra = get_data(processed=4)
    diameter_min = df_muh_kra['diameter'].min()
    diameter_max = df_muh_kra['diameter'].max()
    diameter_km = st.slider("Göktaşının çapı (km)", int(diameter_min), int(diameter_max), 50)

    # diameter_km değerinin df_pred_vel veri setinde mevcut olup olmadığını kontrol etme
    if diameter_km in df_pred_vel['diameter'].values:
        selected_pred_velocity = df_pred_vel[df_pred_vel['diameter'] == diameter_km]['pred_velocity'].values[0]
        velocity_km_s = st.slider("Göktaşının hızı (km/saniye)", 1, 150, int(selected_pred_velocity))
        name = df_pred_vel[df_pred_vel['diameter'] == diameter_km]['name'].values[0]
        st.write("Göktaşının adı:", name)

    else:
        velocity_km_s = st.slider("Göktaşının hızı (1000 km/s)", 1, 150, 20)

    density_kg_m3 = st.slider("Göktaşının yoğunluğu (kg/m^3)", 1, 4000, 2750)

    k = 0.01

    if st.button("Hesapla"):
        mass_kg, impact_energy_joule, impact_energy_kt, crater_radius_m, crater_surface_area_m2 = tabii_efendim(
            diameter_km, velocity_km_s, density_kg_m3, k)
        st.write(f"Çarpma Enerjisi: {impact_energy_joule:.2e} joule ({impact_energy_kt:.2f} kilotons TNT)")

        # Çarpma enerjisinin Hiroşima'ya atılan bomba sayısına denk gelme durumunu hesapla
        # Hiroşima'ya atılan bomba olan Little Boy'un patlaması yaklaşık olarak 15 kiloton TNT enerjisine denk gelir
        # Dolayısıyla, çarpma enerjisinin Hiroşima'ya atılan bomba sayısına denk gelme durumu, çarpma enerjisinin kiloton TNT cinsinden değerini 15 ile bölmekle bulunabilir
        num_little_boys = impact_energy_kt / 15

        st.write(f"Bu çarpma yaklaşık olarak {num_little_boys:.2f} Hiroşima'ya atılan bombaya denk gelir.")

        st.write(f"Krater Yarıçapı: {crater_radius_m / 2:.2f} km")
        result = craterPredict.compare_crater_area_with_countries(crater_surface_area_m2)
        st.write(result)
        st.pyplot(cemberde_krater_ciz(12742, crater_radius_m))


# Diameter Predictions Tab
elif selected_page == "Diameter Tahmini":
    st.title('Asteroid Çap (Diameter) Eksik Verilerinin Tahmini')
    diameter_right, diameter_left = st.columns((1, 1))
    selected_names = ["Hagar", "Hela", "Albert", "Athanasia", "Nicolaia"]
    df_original = get_data(processed=1)
    st.write("")
    if st.button("Çapı Tahmin et"):
        predict_diameter('InsterstellarProject/asteroid/asteroid.csv')
        # Tahminleme sonrası tablo görüntüleme
        diameter_left.subheader("Tahminleme Sonrası")
        df_filled_diameter = get_data(processed=4)
        df_filled_diameter_selected = df_filled_diameter[df_filled_diameter["name"].isin(selected_names)]
        diameter_left.dataframe(df_filled_diameter_selected.loc[:, ["name", "diameter", "data_arc"]])


    # Diameter değeri boş olan 5 örneği ve istenilen isimlere sahip olanları seç

    df_missing_diameter = df_original[
        (df_original["diameter"].isnull()) & (df_original["name"].isin(selected_names))].head()

    # Tahminleme öncesi tablo görüntüleme (belirli sütunlar)
    diameter_right.subheader("Tahminleme Öncesi")
    diameter_right.dataframe(df_missing_diameter.loc[:, ["name", "diameter", "data_arc"]])








# Sun & Asteroids Tab
elif selected_page == "Gunes ve Asteroidler":
    st.session_state['active_page'] = 'Gunes ve Asteroidler'
    if st.session_state['active_page'] == 'Gunes ve Asteroidler':

        st.markdown("""
                <style>
                .title {
                    text-align: center;
                    font-size: 40px;
                    margin-top: 20px;
                    color: #FF4B4B;
                }
                </style>
                <div class="title">3D Distribution of Asteroids in the Solar System</div>
                """, unsafe_allow_html=True)
        # other.title('3D Distribution of Asteroids in the Solar System')
        # df1=get_data()
        # df1 içindekieksim isim değerlerine rastgele isim atayın harf olarak kalıcı olarak kaydedin
        # df1['name'] = df1['name'].apply(lambda x: ''.join(random.choices(string.ascii_letters, k=5)) if pd.isnull(x) else x)
        test = datasetOrbit(plot_title="Gök Cisimlerinin Yörüngeleri", name="Asteroid", fps=10)
        df1 = get_data(processed=0)
        # Styling
        test.faceColor("black")
        test.paneColor("black")
        test.gridColor("#222831")
        test.orbitTransparency(0.5)
        test.labelColor("white")
        test.tickColor("white")

        test.datasetPlotStyle(background_color="dark_background")

        # Sütun ayarları
        test.columnSemiMajorAxis("a")
        test.columnPerihelion("q")
        test.columnEccentricity("e")
        test.columnInclination("i")
        test.columnLongitudeOfAscendingNode("om")
        test.columnArgumentOfPerihelion("w")
        test.columnName("name")

        # Setting axis labels
        test.xLabel("X-Axis")
        test.yLabel("Y-Axis")
        test.zLabel("Z-Axis")


        def refresh_data():
            # Yeni bir örneklem seç
            sample_df = df1.sample(n=10, random_state=42)
            sample_df.to_csv('sample.csv', index=False)

            # Görselleştirme ve animasyonu yeniden oluştur
            test.fileName("sample.csv")
            test.datasetCalculateOrbit(plot_steps=500, n_orbits=12, color="yellow", random_color=True, trajectory=True,
                                       sun=True, delimiter=",")
            test.datasetAnimateOrbit(dpi=150, save=True, export_zoom=3, font_size="xx-small")

            # Animasyonu Streamlit'te göster
            file_path = "Asteroid-orbit.gif"
            with open(file_path, "rb") as file:
                contents = file.read()
                data_url = base64.b64encode(contents).decode("utf-8")

            # other.markdown(centered_html, unsafe_allow_html=True)


        if st.button('Yeni Örneklem ile Görselleştir'):
            refresh_data()


        def get_image_html(file_path):
            with open(file_path, "rb") as file:
                contents = file.read()
                data_url = base64.b64encode(contents).decode("utf-8")
            return f"""
                <div style="display: flex; justify-content: center; align-items: center;">
                    <img src="data:image/gif;base64,{data_url}" alt="Alt Text">
                </div>
                """


        # Görsel için bir yer tutucu oluşturun
        image_placeholder = st.empty()

        # İlk görseli yerleştirin
        image_placeholder.markdown(get_image_html("Asteroid-orbit.gif"), unsafe_allow_html=True)




# main.py






# Ana uygulamanın geri kalanı

elif selected_page == "RFM-ALTV" : # Örnek olarak RFM-ALTV sayfasını seçtim
    st.title('RFM-Asteroid Life Time Value')

    # Veriyi getir
    df_rfm = get_data(processed=4)  # Veriyi get_data fonksiyonu ile çek
    df_cltv = get_data(processed=4)

    # RFM skorlarını hesapla
    last_update_date = pd.to_datetime('2024-04-16')
    segment,df_score = create_rfm(df_rfm, last_update_date)

    # RFM Skorlarının dağılımı için scatter plot oluşturun
    fig = px.scatter(df_score, x='frequency_score', y='monetary_score', color='recency_score',
                     size='rfm_score',  # Recency skorunu boyut olarak kullanabilirsiniz.
                     hover_data=['name', 'recency_score', 'frequency_score', 'monetary_score', 'rfm_score'],  # Detayları göstermek için ek bilgiler
                     title='RFM Skor Dağılımı')
    st.plotly_chart(fig)

    # Segmentasyon2
    #segment_counts = df_score['rfm_score'].value_counts()
    #fig = px.bar(segment_counts, x=segment_counts.index, y=segment_counts.values, title='RFM Skor Segmentasyonu')
    #fig.update_layout(yaxis=dict(type='log'))  # Y ekseni ölçeğini logaritmik yap
    #st.plotly_chart(fig)

    # Segmentasyon tablosu
    #segment_table = df_score['rfm_score'].value_counts().reset_index()
    #segment_table.columns = ['rfm_score', 'Count']
    #st.write(segment_table)

    # RFM skorları tablosunu göster
    #st.write("RFM Skorları:")
    #st.dataframe(df_score[['name', 'frequency_score', 'recency_score', 'monetary_score', 'rfm_score', 'Broad_Segment']].head(5))

    st.write("RFM En Maden Potansiyelleri:")
    sorted_df = df_score[
        ['name', 'frequency_score', 'recency_score', 'monetary_score', 'rfm_score', 'Broad_Segment']].sort_values(
        by='rfm_score', ascending=False)
    st.dataframe(sorted_df.head(5))

    st.write("RFM Yüksek Riskli Operasyonlar :")
    sorted_df = df_score[
        ['name', 'frequency_score', 'recency_score', 'monetary_score', 'rfm_score', 'Broad_Segment']].sort_values(
        by='rfm_score', ascending=True)
    st.dataframe(sorted_df.head(5))

    cltv_result = calculate_cltv(df_cltv)
    st.write(cltv_result[['name', 'diameter', 'data_arc', 'n_obs_used', 'cltv_score']].head(10))



    # Asteroid CLV skorları
    #st.title('Asteroid Customer Lifetime Value (CLV)')
    #df_clv = asteroid_clv_score(df_score, last_update_date)  # Doğru tarihi geçmeyi unutmayın
    #st.dataframe(df_clv[['name', 'recency', 'frequency_score', 'monetary_score', 'rfm_score', 'clv_score']].head())

# RFM analizini uygula
#df_segment, df_score = create_rfm(df_rfm)

