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
# CUSTOM CSS (Tema Asli & Tambahan Label)
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
    .announce-box {
        background-color: #f0f7ff;
        border-left: 5px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        color: #1e3a8a;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================================================
# FUNGSI LOAD EXCEL
# ========================================================================
def load_excel_data(uploaded_file):
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = next((s for s in excel_file.sheet_names if 'hourly' in s.lower() or 'lembar' in s.lower()), excel_file.sheet_names[0])
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=4)
        
        bulan_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        kolom_bulan = [c for c in df.columns if any(b in str(c) for b in bulan_list)]
        
        if kolom_bulan:
            bulan_idx = datetime.now().month - 1
            target_col = kolom_bulan[bulan_idx]
            ghi_values = pd.to_numeric(df.iloc[:24][target_col], errors='coerce').fillna(0).tolist()
            result = pd.DataFrame({'hour': range(24), 'GHI': ghi_values})
            return result, True, f"✅ Data Profil {target_col} Berhasil Dimuat"
    except Exception as e:
        return None, False, f"❌ Error: {str(e)}"
    return None, False, "❌ Format file tidak sesuai"

# ========================================================================
# LOGIKA PREDIKSI & LABELING
# ========================================================================
def get_ghi_quality(ghi):
    if ghi == 0:
        return "🌙 Malam"
    elif ghi < 200:
        return "🔴 Buruk"
    elif 200 <= ghi <= 600:
        return "🟡 Cukup Baik"
    elif ghi > 600:
        return "🟢 Baik"
    return "-"

def predict_with_weather_realtime(historical_data, temp, humidity, pressure, model_type='arima', steps=24):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    base_pattern = historical_data.set_index('hour')['GHI'].to_dict()
    weather_factor = (1 + (temp - 25) * 0.003) * (1 - (humidity / 100) * 0.15) * (pressure / 1013)
    
    results = []
    for i in range(steps + 1):
        future_time = now + timedelta(hours=i)
        f_hour = future_time.hour
        base_ghi = base_pattern.get(f_hour, 0)
        noise = 1 + (np.random.randn() * 0.04) if model_type in ['arima', 'sarima'] else 1.0
        
        if f_hour < 6 or f_hour >= 18:
            final_ghi = 0
        else:
            final_ghi = base_ghi * weather_factor * noise
            
        val_ghi = round(max(0, final_ghi))
        results.append({
            'Waktu': future_time,
            'Jam': future_time.strftime('%H:%M'),
            'GHI (W/m²)': val_ghi,
            'Kualitas': get_ghi_quality(val_ghi),
            'Confidence': f"{max(55, 95 - i*1.6):.1f}%"
        })
    return pd.DataFrame(results)

# ========================================================================
# TAMPILAN UTAMA
# ========================================================================
def main():
    st.markdown("<h1>🌞 Sistem Prediksi GHI di Pulau Jawa</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="announce-box">
            <strong>📢 Informasi Sistem:</strong><br>
            Data profil historis yang akurat untuk lokasi spesifik dapat diperoleh dari 
            <a href="https://globalsolaratlas.info/" target="_blank" style="color: #667eea; font-weight: bold;">Global Solar Atlas</a>. 
            Sistem memproses prediksi 24 jam ke depan secara real-time berdasarkan jam saat ini.
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Pengaturan")
        uploaded_file = st.file_uploader("Upload Data Atlas (Excel)", type=['xlsx'])
        if uploaded_file:
            data, success, message = load_excel_data(uploaded_file)
            if success:
                st.session_state['historical_data'] = data
                st.success(message)
        
        if 'historical_data' not in st.session_state:
            st.session_state['historical_data'] = pd.DataFrame({'hour': range(24), 'GHI': [0]*24})
            st.warning("Gunakan file Solar Atlas untuk hasil nyata")
            
        model_type = st.selectbox("Model Prediksi", ['arima', 'sarima', 'exponential', 'moving_average'])

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Input Kondisi Cuaca")
        with st.form("weather_form"):
            temperature = st.number_input("🌡️ Suhu (°C)", 15.0, 45.0, 30.0)
            humidity = st.number_input("💧 Kelembapan (%)", 0.0, 100.0, 65.0)
            pressure = st.number_input("📊 Tekanan (hPa)", 950.0, 1050.0, 1010.0)
            submitted = st.form_submit_button("🔮 Prediksi 24 Jam Kedepan")
        
        if submitted:
            st.session_state['predictions'] = predict_with_weather_realtime(
                st.session_state['historical_data'], temperature, humidity, pressure, model_type
            )

    with col2:
        if 'predictions' in st.session_state:
            res = st.session_state['predictions']
            pred_now = res.iloc[0]
            pred_1h = res.iloc[1]
            
            st.subheader("📊 Statistik Prediksi Real-Time")
            ms1, ms2, ms3 = st.columns(3)
            with ms1:
                st.metric(f"Saat Ini ({pred_now['Jam']})", f"{pred_now['GHI (W/m²)']} W/m²")
            with ms2:
                diff = int(pred_1h['GHI (W/m²)'] - pred_now['GHI (W/m²)'])
                st.metric(f"Prediksi Jam {pred_1h['Jam']}", f"{pred_1h['GHI (W/m²)']} W/m²", delta=f"{diff} W/m²")
            with ms3:
                st.metric("Puncak Hari Ini", f"{res['GHI (W/m²)'].max()} W/m²")

            st.markdown("---")
            st.subheader("📈 Kurva Proyeksi 24 Jam")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=res['Waktu'], y=res['GHI (W/m²)'],
                mode='lines+markers',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)',
                name='GHI'
            ))
            fig.update_layout(xaxis_title="Waktu", yaxis_title="W/m²", height=350, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Tabel Detail Prediksi (24 Jam Kedepan)")
            # Menampilkan tabel dengan kolom GHI dan Kualitas
            st.dataframe(
                res[['Jam', 'GHI (W/m²)', 'Kualitas', 'Confidence']], 
                use_container_width=True, 
                hide_index=True,
                height=500 
            )
        else:
            st.info("👈 Masukkan data cuaca dan klik tombol 'Prediksi' untuk memulai.")

if __name__ == "__main__":
    main()
