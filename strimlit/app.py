import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# KONFIGURASI HALAMAN (Sesuai Aslinya)
# ========================================================================
st.set_page_config(
    page_title="Sistem Prediksi GHI Pulau Jawa",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================
# CUSTOM CSS (Sesuai Aslinya)
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
    </style>
""", unsafe_allow_html=True)

# ========================================================================
# FUNGSI LOAD EXCEL (DISESUAIKAN UNTUK FORMAT ATLAS)
# ========================================================================
def load_excel_data(uploaded_file):
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        # Cari sheet yang mengandung data profil jam
        sheet_name = next((s for s in excel_file.sheet_names if 'hourly' in s.lower() or 'lembar' in s.lower()), excel_file.sheet_names[0])
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=4)
        
        bulan_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        kolom_bulan = [c for c in df.columns if any(b in str(c) for b in bulan_list)]
        
        if kolom_bulan:
            # Ambil kolom berdasarkan bulan sekarang
            bulan_idx = datetime.now().month - 1
            target_col = kolom_bulan[bulan_idx]
            
            # Ambil 24 jam data
            ghi_values = pd.to_numeric(df.iloc[:24][target_col], errors='coerce').fillna(0).tolist()
            
            # Buat DataFrame historis sederhana untuk pola rata-rata
            result = pd.DataFrame({'hour': range(24), 'GHI': ghi_values})
            return result, True, f"✅ Data Atlas Terdeteksi (Bulan {target_col})"
            
    except Exception as e:
        return None, False, f"❌ Error: {str(e)}"
    return None, False, "❌ Format tidak dikenali."

# ========================================================================
# LOGIKA PREDIKSI REAL-TIME (BARU)
# ========================================================================
def predict_with_weather_realtime(historical_data, temp, humidity, pressure, model_type='arima', steps=24):
    now = datetime.now()
    
    # Pola dasar dari data historis
    base_pattern = historical_data.set_index('hour')['GHI'].to_dict()
    
    # Koreksi cuaca
    weather_factor = (1 + (temp - 25) * 0.003) * (1 - (humidity / 100) * 0.15) * (pressure / 1013)
    
    results = []
    for i in range(steps + 1):
        future_time = now + timedelta(hours=i)
        f_hour = future_time.hour
        
        # Ambil GHI dasar sesuai jam tersebut
        base_ghi = base_pattern.get(f_hour, 0)
        
        # Simulasi variasi model
        noise = 1 + (np.random.randn() * 0.05) if model_type in ['arima', 'sarima'] else 1.0
        
        # Hitung final GHI (Paksa 0 jika malam)
        if f_hour < 6 or f_hour >= 18:
            final_ghi = 0
        else:
            final_ghi = base_ghi * weather_factor * noise
            
        results.append({
            'Waktu': future_time,
            'Jam': future_time.strftime('%H:%M'),
            'GHI': round(max(0, final_ghi)),
            'Confidence': max(60, 95 - i*1.5)
        })
        
    return pd.DataFrame(results)

# ========================================================================
# APLIKASI UTAMA (TAMPILAN ASLI)
# ========================================================================
def main():
    st.markdown("<h1>🌞 Sistem Prediksi GHI di Pulau Jawa</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1em;'>Prediksi Real-Time untuk 24 Jam Ke Depan</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # SIDEBAR (Sesuai Aslinya)
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        uploaded_file = st.file_uploader("Upload File Excel", type=['xlsx', 'xls'])
        
        if uploaded_file:
            data, success, message = load_excel_data(uploaded_file)
            if success:
                st.success(message)
                st.session_state['historical_data'] = data
            else: st.error(message)
        
        if 'historical_data' not in st.session_state:
            # Data fallback jika tidak ada upload
            st.session_state['historical_data'] = pd.DataFrame({'hour': range(24), 'GHI': [0]*24})
            st.info("💡 Silahkan upload data Atlas")
        
        st.markdown("---")
        model_type = st.selectbox("Pilih Model", ['arima', 'sarima', 'exponential', 'moving_average'])

    # MAIN CONTENT (Sesuai Aslinya)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Input Data Cuaca")
        with st.form("weather_form"):
            temperature = st.number_input("🌡️ Suhu (°C)", 15.0, 45.0, 25.0)
            humidity = st.number_input("💧 Kelembapan (%)", 0.0, 100.0, 75.0)
            pressure = st.number_input("📊 Tekanan (hPa)", 950.0, 1050.0, 1006.0)
            submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)
        
        if submitted:
            st.session_state['predictions'] = predict_with_weather_realtime(
                st.session_state['historical_data'], temperature, humidity, pressure, model_type
            )

    with col2:
        if 'predictions' in st.session_state:
            res = st.session_state['predictions']
            pred_now = res.iloc[0]
            pred_1h = res.iloc[1] # Prediksi 1 jam ke depan
            
            # STATISTIK (Sesuai Aslinya dengan update 1 jam kedepan)
            st.subheader("📊 Statistik Prediksi")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("GHI Jam Ini", f"{pred_now['GHI']} W/m²")
            with col_stat2:
                # Menampilkan 1 jam kedepan
                diff = int(pred_1h['GHI'] - pred_now['GHI'])
                st.metric(f"Jam {pred_1h['Jam']}", f"{pred_1h['GHI']} W/m²", delta=f"{diff}")
            with col_stat3:
                st.metric("Puncak Hari Ini", f"{res['GHI'].max()} W/m²")
            with col_stat4:
                st.metric("Estimasi Total", f"{res['GHI'].sum()/1000:.1f} kWh")
            
            st.markdown("---")
            
            # GRAFIK (Sesuai Aslinya)
            st.subheader("📈 Grafik Prediksi GHI")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=res['Waktu'], y=res['GHI'],
                mode='lines+markers',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)',
                name='GHI'
            ))
            fig.update_layout(xaxis_title="Waktu", yaxis_title="W/m²", height=400, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # TABEL (Sesuai Aslinya)
            with st.expander("📋 Tabel Prediksi Detail"):
                st.dataframe(res[['Jam', 'GHI', 'Confidence']], use_container_width=True, hide_index=True)
        else:
            st.info("👈 Masukkan data cuaca dan klik 'Prediksi Sekarang'")

if __name__ == "__main__":
    main()
