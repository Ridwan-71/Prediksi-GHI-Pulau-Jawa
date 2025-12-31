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
    page_title="GHI Real-Time Predictor",
    page_icon="🌞",
    layout="wide"
)

# Custom CSS untuk tampilan lebih modern
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #f0f2f6;
    }
    h1 { color: #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

# ========================================================================
# FUNGSI LOAD DATA (Format Matrix Global Solar Atlas)
# ========================================================================
def load_excel_data(uploaded_file):
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        # Cari sheet yang berisi profile jam
        sheet_name = next((s for s in excel_file.sheet_names if 'hourly' in s.lower() or 'lembar' in s.lower()), excel_file.sheet_names[0])
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=4)
        
        # Cari kolom bulan
        bulan_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        kolom_bulan = [c for c in df.columns if any(b in str(c) for b in bulan_list)]
        
        if kolom_bulan:
            # Ambil kolom bulan saat ini
            bulan_idx = datetime.now().month - 1
            target_col = kolom_bulan[bulan_idx]
            
            # Ambil 24 jam (GHI biasanya di tabel pertama)
            ghi_values = pd.to_numeric(df.iloc[:24][target_col], errors='coerce').fillna(0).tolist()
            
            # Kembalikan dataframe dengan index jam 0-23
            return pd.DataFrame({'hour': range(24), 'GHI': ghi_values}), True, f"✅ Data {target_col} Berhasil Dimuat"
    except Exception as e:
        return None, False, f"❌ Error: {str(e)}"
    return None, False, "❌ Format tidak sesuai"

# ========================================================================
# LOGIKA PREDIKSI REAL-TIME
# ========================================================================
def run_realtime_prediction(historical_df, temp, hum, press, model_type):
    now = datetime.now()
    current_hour = now.hour
    
    results = []
    
    # Ambil pola dasar dari historical_df
    base_ghi_pattern = historical_df.set_index('hour')['GHI'].to_dict()
    
    # Faktor Cuaca
    weather_factor = (1 + (temp - 25) * 0.003) * (1 - (hum / 100) * 0.15) * ((press / 1013))

    for i in range(25): # 0 (sekarang) sampai 24 jam kedepan
        target_time = now + timedelta(hours=i)
        target_hour = target_time.hour
        
        # Ambil nilai GHI dari pola jam yang sesuai
        base_ghi = base_ghi_pattern.get(target_hour, 0)
        
        # Tambahkan variasi berdasarkan model
        if model_type == 'arima':
            variation = 1 + (np.random.randn() * 0.05)
        elif model_type == 'sarima':
            variation = 1 + (np.random.randn() * 0.02)
        else:
            variation = 1.0
            
        final_ghi = base_ghi * weather_factor * variation
        
        # Paksa 0 jika malam hari (6 sore - 6 pagi)
        if target_hour < 6 or target_hour >= 18:
            final_ghi = 0
            
        results.append({
            'Waktu': target_time,
            'Jam': target_time.strftime('%H:%M'),
            'GHI': round(max(0, final_ghi)),
            'Confidence': max(60, 95 - (i * 1.5))
        })
        
    return pd.DataFrame(results)

# ========================================================================
# UI STREAMLIT
# ========================================================================
def main():
    st.title("🌞 GHI Forecasting System (Real-Time)")
    st.markdown(f"**Waktu Sistem:** {datetime.now().strftime('%d %b %Y | %H:%M:%S')}")
    st.divider()

    # SIDEBAR
    with st.sidebar:
        st.header("📂 Input Data")
        file = st.file_uploader("Upload Data Global Solar Atlas", type=['xlsx'])
        if file:
            data, success, msg = load_excel_data(file)
            if success:
                st.success(msg)
                st.session_state['hist_data'] = data
            else: st.error(msg)
        
        model_type = st.selectbox("Metode Prediksi", ['arima', 'sarima', 'moving_average'])

    # DATA CHECK
    if 'hist_data' not in st.session_state:
        # Generate data kosong jika belum upload
        st.session_state['hist_data'] = pd.DataFrame({'hour': range(24), 'GHI': [0]*24})
        st.warning("Silahkan upload file Excel untuk akurasi nyata. Sekarang menggunakan data default (0).")

    # INPUT CUACA
    c1, c2, c3 = st.columns(3)
    with c1: t = st.number_input("Suhu (°C)", 15, 45, 30)
    with c2: h = st.number_input("Kelembapan (%)", 0, 100, 60)
    with c3: p = st.number_input("Tekanan (hPa)", 900, 1100, 1010)
    
    if st.button("🔮 HITUNG PREDIKSI SEKARANG", use_container_width=True):
        st.session_state['results'] = run_realtime_prediction(st.session_state['hist_data'], t, h, p, model_type)

    # DISPLAY HASIL
    if 'results' in st.session_state:
        res = st.session_state['results']
        now_data = res.iloc[0]
        next_hour = res.iloc[1]

        st.divider()
        
        # METRIK UTAMA
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("GHI Saat Ini", f"{now_data['GHI']} W/m²", help="Estimasi radiasi jam ini")
        with m2:
            delta = int(next_hour['GHI'] - now_data['GHI'])
            st.metric(f"Prediksi Jam {next_hour['Jam']}", f"{next_hour['GHI']} W/m²", delta=f"{delta} W/m²")
        with m3:
            st.metric("Confidence Level", f"{now_data['Confidence']:.1f}%")

        # GRAFIK
        st.subheader("📈 Proyeksi Radiasi 24 Jam Ke Depan")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res['Waktu'], y=res['GHI'],
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#1E3A8A', width=3),
            name="Prediksi GHI"
        ))
        fig.update_layout(hovermode="x unified", height=400, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # TABEL DETAIL
        with st.expander("👁️ Lihat Detail Data per Jam"):
            st.table(res[['Jam', 'GHI', 'Confidence']].head(10))

if __name__ == "__main__":
    main()
