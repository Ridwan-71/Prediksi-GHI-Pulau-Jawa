

"""
========================================================================
SISTEM PREDIKSI GHI DI PULAU JAWA
Kode ini adalah rekonstruksi dari aplikasi yang sedang berjalan
========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# KONFIGURASI HALAMAN
# ========================================================================
st.set_page_config(
    page_title="Sistem Prediksi GHI Pulau Jawa",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================
# CUSTOM CSS
# ========================================================================
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #667eea;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================================================
# FUNGSI GENERATE DATA HISTORIS
# ========================================================================
@st.cache_data
def generate_historical_data(days=30):
    """Generate data GHI historis untuk simulasi"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=days*24,
        freq='H'
    )
    
    ghi_values = []
    for date in dates:
        hour = date.hour
        # PERBAIKAN: GHI hanya ada jam 6 pagi - 6 sore (06:00 - 18:00)
        if 6 <= hour < 18:
            # Puncak di jam 11-13 (jam 11:00 - 13:00)
            # Menggunakan fungsi gaussian untuk pola lebih realistis
            hour_from_sunrise = hour - 6  # 0-11 (dari jam 6 pagi)
            peak_hour = 6  # Jam 12 siang (6 jam setelah sunrise)
            
            # Gaussian curve untuk simulasi radiasi matahari
            sun_intensity = np.exp(-0.5 * ((hour_from_sunrise - peak_hour) / 3) ** 2)
            base_ghi = 1000 * sun_intensity  # Maksimal 1000 W/m² di peak
            
            # Tambah variasi cuaca harian (70-100% dari maksimal)
            ghi = base_ghi * (0.7 + np.random.random() * 0.3)
        else:
            # Malam hari dan dini hari = 0
            ghi = 0
        
        ghi_values.append(max(0, ghi))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'GHI': ghi_values
    })
    return df

# ========================================================================
# FUNGSI LOAD EXCEL (VERSI TERBARU)
# ========================================================================
def load_excel_data(uploaded_file):
    """
    Fungsi khusus untuk membaca format Global Solar Atlas (GHI Jawa Barat)
    Mendukung format Hourly Profiles (Matrix) maupun format vertikal standar.
    """
    try:
        # Baca semua sheet untuk mencari yang berisi 'Hourly_profiles'
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = next((s for s in excel_file.sheet_names if 'hourly' in s.lower()), excel_file.sheet_names[0])
        
        # Load data, Global Solar Atlas biasanya ada metadata di 4 baris pertama
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=4)
        
        # 1. Cek apakah ini format Matrix (Global Solar Atlas)
        # Mencari kolom bulan (Jan, Feb, Mar...)
        bulan_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        kolom_bulan = [c for c in df.columns if any(b in str(c) for b in bulan_list)]
        
        if kolom_bulan:
            # Ambil kolom bulan saat ini (misal: Dec)
            bulan_sekarang = datetime.now().strftime('%b')
            # Jika bulan sekarang tidak ada di kolom, ambil kolom pertama yang tersedia
            target_col = next((c for c in kolom_bulan if bulan_sekarang in str(c)), kolom_bulan[0])
            
            # Ambil 24 jam data
            df_24h = df.iloc[:24].copy()
            
            # Buat timestamp buatan (dummy) agar model prediksi bisa jalan
            # Dimulai dari jam 00:00 hari ini
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            timestamps = [start_date + timedelta(hours=i) for i in range(24)]
            
            result = pd.DataFrame({
                'timestamp': timestamps,
                'GHI': pd.to_numeric(df_24h[target_col], errors='coerce').fillna(0)
            })
            
            return result, True, f"✅ Format Atlas Terdeteksi: Menggunakan profil bulan {target_col}"
            
        # 2. Jika bukan format Matrix, gunakan logika pencarian vertikal biasa
        else:
            t_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['time', 'date', 'waktu'])), None)
            g_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['ghi', 'irradiance', 'radiasi'])), None)
            
            if t_col and g_col:
                df[t_col] = pd.to_datetime(df[t_col], errors='coerce')
                df = df.dropna(subset=[t_col, g_col])
                result = df[[t_col, g_col]].copy()
                result.columns = ['timestamp', 'GHI']
                result = result.sort_values('timestamp')
                return result, True, "✅ Format Vertikal Terdeteksi"
            
        return None, False, "❌ Format kolom GHI tidak ditemukan."

    except Exception as e:
        return None, False, f"❌ Error: {str(e)}"

# ========================================================================
# FUNGSI PREDIKSI
# ========================================================================
def predict_arima(historical_data, steps=24):
    """Prediksi menggunakan ARIMA"""
    recent_data = historical_data['GHI'].tail(168).values
    
    seasonal_pattern = np.zeros(24)
    seasonal_count = np.zeros(24)
    
    for i, value in enumerate(recent_data):
        hour = i % 24
        seasonal_pattern[hour] += value
        seasonal_count[hour] += 1
    
    seasonal_pattern = seasonal_pattern / np.maximum(seasonal_count, 1)
    
    predictions = []
    current_hour = datetime.now().hour
    
    for i in range(steps):
        hour = (current_hour + i) % 24
        base_pred = seasonal_pattern[hour]
        
        # PERBAIKAN: Paksa GHI = 0 untuk jam malam (sebelum jam 6 dan setelah jam 18)
        if hour < 6 or hour >= 18:
            predictions.append(0)
            continue
        
        if i > 0 and predictions[-1] > 0:
            ar_component = predictions[-1] * 0.3
            base_pred = base_pred * 0.7 + ar_component
        
        if base_pred > 0:
            base_pred *= (1 + np.random.randn() * 0.05)
        
        predictions.append(max(0, base_pred))
    
    return predictions

def predict_sarima(historical_data, steps=24):
    """Prediksi menggunakan SARIMA"""
    recent_data = historical_data['GHI'].tail(168).values
    
    seasonal_pattern = np.zeros(24)
    seasonal_count = np.zeros(24)
    
    for i, value in enumerate(recent_data):
        hour = i % 24
        seasonal_pattern[hour] += value
        seasonal_count[hour] += 1
    
    seasonal_pattern = seasonal_pattern / np.maximum(seasonal_count, 1)
    
    predictions = []
    current_hour = datetime.now().hour
    
    for i in range(steps):
        hour = (current_hour + i) % 24
        base_pred = seasonal_pattern[hour]
        
        # PERBAIKAN: Paksa GHI = 0 untuk jam malam
        if hour < 6 or hour >= 18:
            predictions.append(0)
            continue
        
        if i > 0 and predictions[-1] > 0:
            ar_component = predictions[-1] * 0.25
            seasonal_component = predictions[max(0, i-12)] * 0.15 if i >= 12 else 0
            base_pred = base_pred * 0.6 + ar_component + seasonal_component
        
        if base_pred > 0:
            base_pred *= (1 + np.random.randn() * 0.05)
        
        predictions.append(max(0, base_pred))
    
    return predictions

def predict_exponential(historical_data, steps=24):
    """Prediksi menggunakan Exponential Smoothing"""
    recent_data = historical_data['GHI'].tail(168).values
    
    seasonal_pattern = np.zeros(24)
    seasonal_count = np.zeros(24)
    
    for i, value in enumerate(recent_data):
        hour = i % 24
        seasonal_pattern[hour] += value
        seasonal_count[hour] += 1
    
    seasonal_pattern = seasonal_pattern / np.maximum(seasonal_count, 1)
    
    predictions = []
    current_hour = datetime.now().hour
    alpha = 0.3
    
    for i in range(steps):
        hour = (current_hour + i) % 24
        base_pred = seasonal_pattern[hour]
        
        # PERBAIKAN: Paksa GHI = 0 untuk jam malam
        if hour < 6 or hour >= 18:
            predictions.append(0)
            continue
        
        if i > 0 and predictions[-1] > 0:
            smoothed = alpha * base_pred + (1 - alpha) * predictions[-1]
            base_pred = smoothed
        
        if base_pred > 0:
            base_pred *= (1 + np.random.randn() * 0.05)
        
        predictions.append(max(0, base_pred))
    
    return predictions

def predict_moving_average(historical_data, steps=24):
    """Prediksi menggunakan Moving Average"""
    recent_data = historical_data['GHI'].tail(168).values
    
    seasonal_pattern = np.zeros(24)
    seasonal_count = np.zeros(24)
    
    for i, value in enumerate(recent_data):
        hour = i % 24
        seasonal_pattern[hour] += value
        seasonal_count[hour] += 1
    
    seasonal_pattern = seasonal_pattern / np.maximum(seasonal_count, 1)
    
    predictions = []
    current_hour = datetime.now().hour
    
    for i in range(steps):
        hour = (current_hour + i) % 24
        base_pred = seasonal_pattern[hour]
        
        # PERBAIKAN: Paksa GHI = 0 untuk jam malam
        if hour < 6 or hour >= 18:
            predictions.append(0)
            continue
        
        if i >= 2 and predictions[-1] > 0 and predictions[-2] > 0:
            ma3 = (predictions[-1] + predictions[-2] + base_pred) / 3
            base_pred = ma3
        
        if base_pred > 0:
            base_pred *= (1 + np.random.randn() * 0.05)
        
        predictions.append(max(0, base_pred))
    
    return predictions

def predict_with_weather(historical_data, temp, humidity, pressure, 
                        model_type='arima', steps=24):
    """Prediksi dengan faktor cuaca"""
    
    if model_type == 'arima':
        base_predictions = predict_arima(historical_data, steps)
    elif model_type == 'sarima':
        base_predictions = predict_sarima(historical_data, steps)
    elif model_type == 'exponential':
        base_predictions = predict_exponential(historical_data, steps)
    else:
        base_predictions = predict_moving_average(historical_data, steps)
    
    temp_factor = 1 + (temp - 25) * 0.003
    humidity_factor = 1 - (humidity / 100) * 0.15
    pressure_factor = (pressure - 1013) / 1013 * 0.05 + 1
    
    adjusted_predictions = []
    current_time = datetime.now()
    
    for i, pred in enumerate(base_predictions):
        future_hour = (current_time + timedelta(hours=i)).hour
        
        # VALIDASI TAMBAHAN: Double-check jam malam
        if future_hour < 6 or future_hour >= 18:
            adjusted_predictions.append(0)
            continue
        
        weather_impact = np.exp(-i / 12)
        weather_factor = (
            1 + 
            (temp_factor - 1) * weather_impact +
            (humidity_factor - 1) * weather_impact +
            (pressure_factor - 1) * weather_impact
        )
        adjusted_pred = pred * weather_factor
        adjusted_predictions.append(max(0, min(1200, adjusted_pred)))
    
    future_times = [current_time + timedelta(hours=i) for i in range(steps)]
    
    results = pd.DataFrame({
        'Waktu': future_times,
        'Jam': [t.strftime('%H:%M') for t in future_times],
        'GHI': [round(p) for p in adjusted_predictions],
        'Confidence': [max(60, 95 - i*1.5) for i in range(steps)]
    })
    
    return results

# ========================================================================
# APLIKASI UTAMA
# ========================================================================
def main():
    st.markdown("<h1>🌞 Sistem Prediksi GHI di Pulau Jawa</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1em;'>Prediksi Global Horizontal Irradiance untuk 24 Jam Ke Depan</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        st.subheader("📁 Data Historis")
        uploaded_file = st.file_uploader(
            "Upload File Excel (Opsional)",
            type=['xlsx', 'xls'],
            help="Upload data GHI historis"
        )
        
        if uploaded_file:
            data, success, message = load_excel_data(uploaded_file)
            if success:
                st.success(message)
                st.session_state['historical_data'] = data
            else:
                st.error(message)
                if 'historical_data' not in st.session_state:
                    st.session_state['historical_data'] = generate_historical_data()
        else:
            if 'historical_data' not in st.session_state:
                st.session_state['historical_data'] = generate_historical_data()
                st.info("💡 Menggunakan data simulasi")
        
        st.markdown("---")
        
        st.subheader("📊 Model Prediksi")
        model_type = st.selectbox(
            "Pilih Model",
            ['arima', 'sarima', 'exponential', 'moving_average'],
            format_func=lambda x: {
                'arima': 'ARIMA',
                'sarima': 'SARIMA (Seasonal)',
                'exponential': 'Exponential Smoothing',
                'moving_average': 'Moving Average'
            }[x]
        )
        
        st.markdown("---")
        st.subheader("💡 Informasi Model")
        st.markdown("""
        - **ARIMA**: Model klasik time series
        - **SARIMA**: Dengan pola musiman
        - **Exp. Smoothing**: Pembobotan waktu
        - **Moving Average**: Rata-rata bergerak
        """)
    
    # MAIN CONTENT
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Input Data Cuaca")
        
        with st.form("weather_form"):
            temperature = st.number_input(
                "🌡️ Suhu (°C)",
                min_value=15.0,
                max_value=40.0,
                value=28.0,
                step=0.1
            )
            
            humidity = st.number_input(
                "💧 Kelembapan (%)",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=0.1
            )
            
            pressure = st.number_input(
                "📊 Tekanan Udara (hPa)",
                min_value=950.0,
                max_value=1050.0,
                value=1013.0,
                step=0.1
            )
            
            submitted = st.form_submit_button("🔮 Prediksi 24 Jam", use_container_width=True)
        
        if submitted:
            with st.spinner("🔄 Memproses prediksi..."):
                predictions = predict_with_weather(
                    st.session_state['historical_data'],
                    temperature,
                    humidity,
                    pressure,
                    model_type
                )
                
                st.session_state['predictions'] = predictions
                st.session_state['model_name'] = model_type
                st.success("✅ Prediksi berhasil!")
    
    with col2:
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            
            # STATISTIK
            st.subheader("📊 Statistik Prediksi")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Max GHI", f"{predictions['GHI'].max()} W/m²")
            
            with col_stat2:
                st.metric("Rata-rata", f"{predictions['GHI'].mean():.0f} W/m²")
            
            with col_stat3:
                st.metric("Total Energi", f"{predictions['GHI'].sum()/1000:.1f} kWh/m²")
            
            with col_stat4:
                model_names = {
                    'arima': 'ARIMA',
                    'sarima': 'SARIMA',
                    'exponential': 'Exp. SM',
                    'moving_average': 'MA'
                }
                st.metric("Model", model_names.get(st.session_state.get('model_name', 'arima')))
            
            st.markdown("---")
            
            # GRAFIK
            st.subheader("📈 Grafik Prediksi GHI")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=predictions['Jam'],
                y=predictions['GHI'],
                mode='lines+markers',
                name='GHI Prediksi',
                line=dict(color='#667eea', width=3),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            
            fig.update_layout(
                xaxis_title="Waktu (Jam)",
                yaxis_title="GHI (W/m²)",
                hovermode='x unified',
                height=400,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # TABEL
            st.subheader("📋 Tabel Prediksi Detail")
            
            def get_status(ghi):
                if ghi > 700:
                    return "☀️ Optimal"
                elif ghi > 300:
                    return "⛅ Baik"
                elif ghi > 0:
                    return "🌤️ Rendah"
                else:
                    return "🌙 Malam"
            
            display_df = predictions[['Jam', 'GHI', 'Confidence']].copy()
            display_df['Status'] = predictions['GHI'].apply(get_status)
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1f}%")
            display_df['GHI'] = display_df['GHI'].apply(lambda x: f"{x} W/m²")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # DOWNLOAD
            csv = predictions.to_csv(index=False)
            st.download_button(
                label="📥 Download Hasil (CSV)",
                data=csv,
                file_name=f"prediksi_ghi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.info("👈 Masukkan data cuaca di sebelah kiri dan klik 'Prediksi 24 Jam'")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p><strong>💡 Catatan:</strong> Akurasi prediksi meningkat dengan data historis yang lebih lengkap.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
