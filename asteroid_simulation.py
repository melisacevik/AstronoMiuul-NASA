import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

class CustomPage:
    def __init__(self, page_name):
        self.page_name = page_name
        self.sidebar = st.sidebar

    def write(self, *args, **kwargs):
        st.write(*args, **kwargs)

    def markdown(self, *args, **kwargs):
        st.markdown(*args, **kwargs)

    def success(self, *args, **kwargs):
        st.success(*args, **kwargs)

    def is_active_page(self):
        """Mevcut sayfanın aktif olup olmadığını kontrol eder."""
        return st.session_state.get('active_page', '') == self.page_name

def asteroid_orbit_simulation(page, df):
    if page.is_active_page():
        page.sidebar.write(f'{page.page_name} - Asteroid Yörünge Parametreleri 🌌')

        # Veri setinden isim seçeneklerini al
        names = df['name'].unique()

        # İsim seçim kutusu
        selected_name = page.sidebar.selectbox('Asteroid Seçiniz', names)

        # Seçilen ismin eksantriklik ve yarı-büyük eksen değerlerini al
        selected_row = df[df['name'] == selected_name].iloc[0]
        eccentricity = selected_row['e']
        semi_major_axis = selected_row['a']

        # Giriş alanları
        semi_major_axis = page.sidebar.number_input('Yarı-Büyük Eksen (au)', value=semi_major_axis, format="%.3f")
        eccentricity = page.sidebar.number_input('Eksantriklik', min_value=0.0, max_value=1.0, value=eccentricity, step=0.01)

        def calculate_min_distance_to_earth(semi_major_axis, eccentricity):
            """Dünya'ya olan minimum mesafeyi hesaplar."""
            if semi_major_axis < 1.1 and eccentricity < 0.1:
                return np.random.uniform(0.05, 0.1)  # Tehlikeli yakınlaşma
            else:
                return np.random.uniform(0.1, 5.0)  # Rastgele bir değer döndür

        # Simülasyon başlatma butonu
        if page.sidebar.button('Simülasyonu Başlat 🚀'):
            min_distance = calculate_min_distance_to_earth(semi_major_axis, eccentricity)
            km_distance = min_distance * 146000000
            page.markdown(f'### Simülasyon Sonucu 📊')
            page.markdown(f'Dünya\'ya olan minimum yaklaşma mesafesi: **{min_distance:.2f} au** , {km_distance:.2f} km**')

            if min_distance < 0.1:
                # Tehlikeli yakınlaşma uyarıları
                page.markdown(f"""
                    <audio autoplay>
                    <source src="https://raw.githubusercontent.com/KuleGizem/asteroid/main/military-alarm-129017.mp3" type="audio/mp3">
                    Your browser does not support the audio element.
                    </audio>
                    """, unsafe_allow_html=True)
                page.markdown("""
                    <style>
                    @keyframes blinker {  
                        50% { background-color: #ff0000; }
                    }
                    .blinking-alert {
                        animation: blinker 1s linear infinite;
                        color: white;
                        font-weight: bold;
                        padding: 10px;
                        text-align: center;
                    }
                    </style>
                    <div class="blinking-alert">⚠️ Tehlikeli Yakınlaşma!</div>
                    """, unsafe_allow_html=True)
            else:
                page.success('✅ Güvenli Mesafe.')

            # Yörünge eksenini çiz
            draw_orbit(semi_major_axis, eccentricity)

def draw_orbit(semi_major_axis, eccentricity):
    # Yörünge için theta değerlerini oluştur
    theta = np.linspace(0, 2 * np.pi, 1000)
    # Yörünge eğrisinin parametrik denklemi
    r = semi_major_axis * (1 - eccentricity ** 2) / (1 + eccentricity * np.cos(theta))

    # Polar koordinatlarda x ve y koordinatlarını hesapla
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Dünya'yı temsil eden bir nokta ekleyin
    earth_x = 1.0  # Örneğin, Dünya'nın x koordinatı
    earth_y = 0.0  # Örneğin, Dünya'nın y koordinatı

    # Çizimi oluştur
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue', label='Asteroid Yörünge')
    ax.scatter(earth_x, earth_y, color='green', marker='o', label='Dünya')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title('Asteroid Yörünge Eğrisi')
    ax.set_xlabel('X (au)')
    ax.set_ylabel('Y (au)')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend()
    st.pyplot(fig)

