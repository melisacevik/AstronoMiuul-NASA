import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from difrfm import create_rfm
from datasetOrbit import datasetOrbit
import base64
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import craterPredict

# Set page config
st.set_page_config(layout="wide", page_title="Asteroid", page_icon="🌠")


def generate_stars_css(num_stars=300):
    stars_css = ""
    for _ in range(num_stars):
        x = random.randint(0, 100)  # Viewport width percentage
        y = random.randint(0, 100)  # Viewport height percentage
        stars_css += f"{x}vw {y}vh #FFF, "
    stars_css = stars_css.rstrip(", ")
    return f".star {{ width: 1px; height: 1px; background: transparent; box-shadow: {stars_css}; position: fixed; }}"

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
@st.cache_data
def get_data():
    data = pd.read_csv('InsterstellarProject/asteroid/asteroid.csv')
    data['diameter'] = pd.to_numeric(data['diameter'], errors='coerce') # diameter sütununu sayısal bir türe dönüştürme
    data = data.dropna(subset=['diameter']) # NaN değerlere sahip satırları kaldırma
    return data


# Define functions
def tabii_efendim(diameter_km, velocity_km_s=100, density_kg_m3=300000, k=0.0001):
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
    other.markdown(centered_html, unsafe_allow_html=True)

def cemberde_krater_ciz(dunya_cap_m, krater_yaricap_m):
    """
    Dünya üzerinde bir kraterin çizimini yapar.
    """
    fig, ax = plt.subplots()

    # Dünya'yı temsil eden çember
    dunya = Circle((0.5, 0.5), 0.4, color='blue', label='Dünya')
    ax.add_artist(dunya)

    # Krateri temsil eden çember
    # Kraterin yeri ve boyutu semboliktir.
    krater = Circle((0.5, 0.5), krater_yaricap_m / dunya_cap_m * 0.4, color='red', label='Krater')
    ax.add_artist(krater)

    # Grafiği düzenleme
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')  # Eksenleri gizleme
    ax.legend()  # Efsane eklemek için

    return fig

# Load data
df = get_data()

# Define layout
st.sidebar.title('Asteroid Projesi')
selected_page = st.sidebar.radio("Gitmek istediğiniz sayfayı seçin", ["Hosgeldiniz", "Asteroid Madencilik Potansiyeli", "Asteroid Carpma Senaryoları", "Muhtemel Krater", "Main", "Other", "Eksikler", "RFM-CLTV"])

# Hosgeldiniz Tab
if selected_page == "Hosgeldiniz":
    st.title('Hosgeldiniz')
    st.write("""
        Bu sekmede, Dünya'ya yakın geçiş yapan asteroidler hakkında ilginç bilgiler bulabilirsiniz.
    """)
    close_approach_df = df[df['moid'] < 0.05]  # Örneğin 0.05 AU'den daha yakın geçişler
    fig = px.scatter(close_approach_df, x='moid', y='diameter', color='class', size='diameter', hover_data=['name'],
                     title='Dünyaya Yakın Geçiş Yapan Asteroidler')
    st.plotly_chart(fig)

# Asteroid Madencilik Potansiyeli Tab
elif selected_page == "Asteroid Madencilik Potansiyeli":
    st.title('Asteroid Madencilik Potansiyeli')
    st.write("""
        - **Asteroidler**: Değerli metalleri büyük miktarlarda barındırır.
        - **Su**: Uzayda suyun önemi, yaşamın temeli olmasının yanı sıra, uzay araçları için yakıt olarak da kullanılabilir.
    """)
    valuable_asteroids = df[df['diameter'] > 1]  # Örnek bir filtre
    fig = px.scatter(valuable_asteroids, x='a', y='diameter', color='albedo', title='Madencilik Potansiyeli Yüksek Asteroidler')
    st.plotly_chart(fig)

# Asteroid Carpma Senaryoları Tab
elif selected_page == "Asteroid Carpma Senaryoları":
    st.title('Asteroid Çarpma Senaryoları')
    st.write("""
        - **Küçük Asteroidler**: Atmosferde yanarak zararsız hale gelir.
        - **Büyük Asteroidler**: Küresel çapta felaketlere yol açabilir.
    """)
    impact_scenarios = df.sample(n=5)  # Örnek bir seçim
    fig = px.scatter(impact_scenarios, x='diameter', y='moid', size='diameter', color='condition_code', title='Asteroid Çarpma Senaryoları')
    st.plotly_chart(fig)

# Muhtemel Krater Tab
elif selected_page == "Muhtemel Krater":
    st.title('Muhtemel Krater')
    st.write("""
        Bu sekmede, bir göktaşının çarpması sonucu oluşacak kraterin tahmini boyutunu hesaplayabilirsiniz.
    """)
    diameter_min = df['diameter'].min()
    diameter_max = df['diameter'].max()
    diameter_km = st.slider("Göktaşının çapı (km)", int(diameter_min), int(diameter_max), 10)
    velocity_km_s = st.slider("Göktaşının hızı (km/s)", 1, 150, 20)
    density_kg_m3 = st.slider("Göktaşının yoğunluğu (kg/m^3)", 1, 1000, 5)
    k = 0.0001

    if st.button("Hesapla"):
        mass_kg, impact_energy_joule, impact_energy_kt, crater_radius_m, crater_surface_area_m2 = tabii_efendim(diameter_km, velocity_km_s, density_kg_m3, k)
        st.write(f"Göktaşının kütlesi: {mass_kg:.2e} kg")
        st.write(f"Çarpma enerjisi: {impact_energy_joule:.2e} joule ({impact_energy_kt:.2f} kiloton TNT)")
        st.write(f"Krater yarıçapı: {crater_radius_m:.2f} metre")
        st.write(f"Krater yüzey alanı: {crater_surface_area_m2 / 1e6:.2f} metrekare")  # Metrekareyi kilometrekareye dönüştürme
        crm2 = crater_radius_m / 1000000
        result = craterPredict.compare_crater_area_with_countries(crm2)
        st.write(result)
        st.pyplot(cemberde_krater_ciz(12742, crater_radius_m))

# Main Tab
elif selected_page == "Main":
    st.title('Main')
    df = get_data()  # Veriyi get_data fonksiyonu ile çek
    st.write("Veri Setinin İlk Beş Satırı:")
    st.dataframe(df.head())

# Other Tab
elif selected_page == "Other":
    st.title('Other')
    # Diğer içeriği buraya ekleyin

# Eksikler Tab
elif selected_page == "Eksikler":
    st.title('Eksikler')
    # Eksik veri analizi içeriğini buraya ekleyin
    eksik_veri_tablosu = eksik_veri_detay(df)
    st.write("Eksik Veri Detayları:")
    st.write(eksik_veri_tablosu)

# RFM-CLTV Tab
elif selected_page == "RFM-CLTV":
    st.title('RFM-CLTV')
    # RFM-CLTV analizi içeriğini buraya ekleyin
    # RFM Skorlarının dağılımı,
    fig = px.scatter(sample_segments, x='n_obs_used', y='H', color='RFM_Score', title='RFM Skorlarının Dağılımı')
    rfm_cltv_left_panel.plotly_chart(fig)

    # Segmentasyon2

    segment_counts = sample_segments['Broad_Segment'].value_counts()
    fig = px.bar(segment_counts, x=segment_counts.index, y=segment_counts.values, title='Segmentasyon2')

    # Y ekseni ölçeğini ayarla
    fig.update_layout(yaxis=dict(type='log'))

    rfm_cltv_left_panel.plotly_chart(fig)

    # Segmentasyon tablosu
    segment_table = sample_segments['Broad_Segment'].value_counts().reset_index()
    segment_table.columns = ['Broad_Segment', 'Count']
    rfm_cltv_left_panel.write(segment_table)
