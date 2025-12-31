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
    page_title="Sistem Prediksi GHI Pulau Jawa - Real Time",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================
# CUSTOM CSS
# ========================================================================
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    h1 { color: #667eea; text-align: center; padding: 1rem 0; }
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
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================================================
# FUNGSI GENERATE DATA HISTORIS (SIMULASI)
# ========================================================================
@st.cache_data
def generate_historical_data(days=30):
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=days*24,
        freq='H'
    )
    ghi_values = []
    for date in dates:
        hour = date.hour
        if 6 <= hour < 18:
            sun_intensity = np.exp(-0.5 * ((hour - 12) / 3) ** 2)
            base_ghi = 1000 * sun_intensity
            ghi = base_ghi * (0.7 + np.random.random() * 0.3)
        else:
            ghi = 0
        ghi_values.append(max(0, ghi))
    return pd.DataFrame({'timestamp': dates, 'GHI': ghi_values})

# ========================================================================
# FUNGSI LOAD EXCEL (OPTIMASI UNTUK GLOBAL SOLAR ATLAS)
# ========================================================================
def load_excel_data(uploaded_file):
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = next((s for s in excel_file.sheet_names if 'hourly' in s.lower()), excel_file.sheet_names[0])
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=4)
        
        bulan_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        kolom_bulan = [c for c in df.columns if any(b in str(c) for b in bulan_list)]
        
        if kolom_bulan:
            bulan_sekarang = datetime.now().strftime('%b')
            target_col = next((c for c in kolom_bulan if bulan_sekarang in str(c)), kolom_bulan[0])
            df_24h = df.iloc[:24].copy()
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            timestamps = [start_date + timedelta(hours=i) for i in range(24)]
            
            result = pd.DataFrame({
                'timestamp': timestamps,
                'GHI': pd.to_numeric(df_24h[target_col], errors='coerce').fillna(0)
            })
            return result, True, f"✅ Format Atlas Terdeteksi (Bulan {target_col})"
        else:
            t_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['time', 'date', 'waktu'])), None)
            g_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['ghi', 'irradiance', 'radiasi'])), None)
            if t_col and g_col:
                df[t_col] = pd.to_datetime(df[t_col], errors='coerce')
                df = df.dropna(subset=[t_col, g_col])
                result = df[[t_col, g_col]].copy()
                result.columns = ['timestamp', 'GHI']
                return result, True, "✅ Format Vertikal Terdeteksi"
        return None, False, "❌ Format tidak dikenali."
    except Exception as e:
        return None, False, f"❌ Error: {str(e)}"

# ========================================================================
# LOGIKA PREDIKSI (SINKRON DENGAN WAKTU SEKARANG)
# ========================================================================

def get_base_prediction(historical_data, model_type, steps):
    # Mengambil pola rata-rata per jam dari data historis
    recent_data = historical_data.tail(168).copy()
    recent_data['hour'] = recent_data['timestamp'].dt.hour
    seasonal_pattern = recent_data.groupby('hour')['GHI'].mean().reindex(range(24), fill_value=0).values
    
    predictions = []
    current_hour = datetime.now().hour
    
    for i in range(steps):
        hour = (current_hour + i) % 24
        base_pred = seasonal_pattern[hour]
        
        # Logika sederhana Model
        if model_type == 'arima':
            noise = np.random.randn() * 0.05
        elif model_type == 'sarima':
            noise = np.random.randn() * 0.03
        else:
            noise = 0
            
        final_base = base_pred * (1 + noise)
        predictions.append(max(0, final_base))
        
    return predictions

def predict_with_weather(historical_data, temp, humidity, pressure, model_type='arima', steps=24):
    """Prediksi Real-Time: Jam sekarang s/d 24 jam kedepan"""
    
    # +1 step untuk mendapatkan data 'saat ini' dan '24 jam kedepan'
    base_preds = get_base_prediction(historical_data, model_type, steps + 1)
    
    temp_factor = 1 + (temp - 25) * 0.003
    humidity_factor = 1 - (humidity / 100) * 0.15
    pressure_factor = (pressure - 1013) / 1013 * 0.05 + 1
    
    results = []
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    for i, pred in enumerate(base_preds):
        target_time = now + timedelta(hours=i)
        hour = target_time.hour
        
        # Paksa 0 jika malam hari
        if hour < 6 or hour >= 18:
            final_ghi = 0
        else:
            # Dampak cuaca berkurang seiring jauhnya prediksi (decay)
            weather_impact = np.exp(-i / 12)
            factor = 1 + ((temp_factor-1) + (humidity_factor-1) + (pressure_factor-1)) * weather_impact
            final_ghi = pred * factor
            
        results.append({
            'Waktu': target_time,
            'Jam': target_time.strftime('%H:%M'),
            'GHI': round(max(0, min(1200, final_ghi))),
            'Confidence': max(60, 95 - i*1.5)
        })
    
    return pd.DataFrame(results)

# ========================================================================
# UI UTAMA
# ========================================================================
def main():
    st.markdown("<h1>🌞 Sistem Prediksi GHI Real-Time</h1>", unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        uploaded_file = st.file_uploader("Upload File (Atlas/Excel)", type=['xlsx', 'xls'])
        
        if uploaded_file:
            data, success, msg = load_excel_data(uploaded_file)
            if success:
                st.success(msg)
                st.session_state['historical_data'] = data
            else: st.error(msg)
        
        if 'historical_data' not in st.session_state:
            st.session_state['historical_data'] = generate_historical_data()
            st.info("💡 Menggunakan data simulasi")
            
        model_type = st.selectbox("Model Prediksi", ['arima', 'sarima', 'exponential', 'moving_average'])

    # LAYOUT
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Kondisi Cuaca Saat Ini")
        with st.form("weather_form"):
            t = st.number_input("🌡️ Suhu (°C)", 15.0, 45.0, 28.0)
            h = st.number_input("💧 Kelembapan (%)", 0.0, 100.0, 70.0)
            p = st.number_input("📊 Tekanan (hPa)", 900.0, 1100.0, 1010.0)
            submitted = st.form_submit_button("🔮 Jalankan Prediksi Real-Time")
            
        if submitted:
            st.session_state['predictions'] = predict_with_weather(
                st.session_state['historical_data'], t, h, p, model_type
            )

    with col2:
        if 'predictions' in st.session_state:
            res = st.session_state['predictions']
            # Data real-time (jam sekarang dan 1 jam lagi)
            ghi_now = res.iloc[0]
            ghi_1h = res.iloc[1]
            
            st.subheader("⚡ Insight Prediksi")
            
            # Baris Metrik Real-Time
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(f"Saat Ini ({ghi_now['Jam']})", f"{ghi_now['GHI']} W/m²")
            with m2:
                # PREDIKSI 1 JAM KEDEPAN
                st.metric(f"Prediksi 1 Jam ({ghi_1h['Jam']})", f"{ghi_1h['GHI']} W/m²", 
                          delta=f"{ghi_1h['GHI'] - ghi_now['GHI']} dari skrg")
            with m3:
                st.metric("Puncak Hari Ini", f"{res['GHI'].max()} W/m²")

            # GRAFIK 24 JAM
            st.markdown("---")
            st.subheader("📈 Proyeksi 24 Jam Ke Depan")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['Waktu'], y=res['GHI'], mode='lines+markers',
                                     fill='tozeroy', line=dict(color='#667eea', width=3),
                                     name="Prediksi GHI"))
            fig.update_layout(xaxis_title="Waktu", yaxis_title="GHI (W/m²)", height=350,
                              margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # TABEL DETAIL
            with st.expander("📋 Lihat Tabel Detail Prediksi"):
                st.dataframe(res, use_container_width=True, hide_index=True)
        else:
            st.info("👈 Klik tombol 'Jalankan Prediksi' untuk melihat hasil.")

if __name__ == "__main__":
    main()
