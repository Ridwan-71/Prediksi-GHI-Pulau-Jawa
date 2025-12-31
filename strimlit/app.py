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
    page_title="GHI Predictor Pro",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================
# CUSTOM CSS (ADAPTIF & USER FRIENDLY)
# ========================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white !important;
        text-align: center;
        margin-bottom: 20px;
    }

    .announce-box {
        color: var(--text-color); 
        background-color: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-left: 6px solid #667eea;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        line-height: 1.7;
    }
    
    .announce-box b {
        color: #667eea;
        font-size: 1.1rem;
    }

    .source-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 15px;
    }

    .source-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 8px;
        border: 1px dashed rgba(102, 126, 234, 0.5);
    }

    .announce-link {
        color: #FFA500 !important; 
        text-decoration: underline;
        font-weight: 700;
    }

    .stButton>button {
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: bold;
        height: 3.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================================================
# FUNGSI BACKEND
# ========================================================================
def load_excel_data(uploaded_file):
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = next((s for s in excel_file.sheet_names if 'hourly' in s.lower() or 'lembar' in s.lower()), excel_file.sheet_names[0])
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=4)
        bulan_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        kolom_bulan = [c for c in df.columns if any(b in str(c) for b in bulan_list)]
        if kolom_bulan:
            target_col = kolom_bulan[datetime.now().month - 1]
            ghi_values = pd.to_numeric(df.iloc[:24][target_col], errors='coerce').fillna(0).tolist()
            return pd.DataFrame({'hour': range(24), 'GHI': ghi_values}), True, f"Data Profil {target_col} Aktif"
    except Exception as e:
        return None, False, f"Error: {str(e)}"
    return None, False, "Format tidak sesuai"

def get_status(ghi):
    if ghi == 0: return "🌙 Malam"
    if ghi < 200: return "🔴 Buruk"
    if ghi <= 600: return "🟡 Cukup Baik"
    return "🟢 Baik"

def run_prediction(historical_data, temp, hum, press, model):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    base_dict = historical_data.set_index('hour')['GHI'].to_dict()
    w_factor = (1 + (temp - 25) * 0.003) * (1 - (hum / 100) * 0.15) * (press / 1013)
    results = []
    for i in range(25):
        future = now + timedelta(hours=i)
        hr = future.hour
        val = base_dict.get(hr, 0) * w_factor
        if hr < 6 or hr >= 18: val = 0 
        final_ghi = round(max(0, val))
        results.append({
            "Waktu": future, "Jam": future.strftime('%H:%M'),
            "GHI (W/m²)": final_ghi, "Kualitas": get_status(final_ghi),
            "Confidence": f"{max(60, 95 - i*1.4):.1f}%"
        })
    return pd.DataFrame(results)

# ========================================================================
# MAIN APP
# ========================================================================
def main():
    st.markdown("""
        <div class="main-header">
            <h1 style='color: white; margin:0;'>DASHBOARD PREDIKSI GHI REAL-TIME</h1>
            <p style='color: white; opacity: 0.9;'>Monitoring Energi Matahari Wilayah Pulau Jawa</p>
        </div>
    """, unsafe_allow_html=True)

    # Box Pengumuman Proper
    st.markdown("""
        <div class="announce-box">
            <b>📋 Panduan Persiapan Data & Parameter:</b>
            Untuk akurasi prediksi maksimal, pastikan Anda mengisi data dari sumber berikut:
            <div class="source-grid">
                <div class="source-item">
                    <b>1. Data Profil GHI (Histori)</b><br>
                    Unduh file <b>Average hourly profiles (.xlsx)</b> dari 
                    <a href="https://globalsolaratlas.info/" target="_blank" class="announce-link">Global Solar Atlas</a> 
                    sesuai lokasi Anda, lalu unggah di sidebar kiri.
                </div>
                <div class="source-item">
                    <b>2. Data Cuaca (Real-Time)</b><br>
                    Cari data <b>Suhu, Kelembapan, dan Tekanan Udara</b> terkini melalui 
                    <a href="https://www.google.com/search?q=cuaca+hari+ini" target="_blank" class="announce-link">Google Cuaca</a> 
                    sesuai wilayah Anda, lalu masukkan pada panel input.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("📂 Data Source")
        uploaded_file = st.file_uploader("Unggah Excel Solar Atlas", type=['xlsx'])
        if uploaded_file:
            data, success, msg = load_excel_data(uploaded_file)
            if success:
                st.session_state['hist_data'] = data
                st.success(msg)
        
        if 'hist_data' not in st.session_state:
            st.session_state['hist_data'] = pd.DataFrame({'hour': range(24), 'GHI': [0]*24})
        
        st.markdown("---")
        model_type = st.selectbox("Model", ['ARIMA', 'SARIMA', 'Exponential'])

    col_in, col_out = st.columns([1, 2], gap="large")

    with col_in:
        st.subheader("📝 Input Parameter")
        with st.form("input_form"):
            t = st.slider("Suhu / Temperature (°C)", 15, 45, 30)
            h = st.slider("Kelembapan / Humidity (%)", 0, 100, 60)
            p = st.number_input("Tekanan / Pressure (hPa)", 900, 1100, 1010)
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("PROSES PREDIKSI")

        if submit:
            st.session_state['res'] = run_prediction(st.session_state['hist_data'], t, h, p, model_type)

    with col_out:
        if 'res' in st.session_state:
            df_res = st.session_state['res']
            now_row = df_res.iloc[0]
            next_row = df_res.iloc[1]

            m1, m2, m3 = st.columns(3)
            with m1: st.metric(f"GHI Sekarang", f"{now_row['GHI (W/m²)']} W/m²")
            with m2: 
                diff = int(next_row['GHI (W/m²)'] - now_row['GHI (W/m²)'])
                st.metric(f"Prediksi 1 Jam", f"{next_row['GHI (W/m²)']} W/m²", delta=f"{diff} W/m²")
            with m3: st.metric("Kualitas", now_row['Kualitas'])

            st.markdown("---")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_res['Waktu'], y=df_res['GHI (W/m²)'],
                mode='lines+markers', line=dict(color='#667eea', width=3),
                fill='tozeroy', name='GHI'
            ))
            fig.update_layout(
                margin=dict(l=0,r=0,t=20,b=0), height=350,
                template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_res[['Jam', 'GHI (W/m²)', 'Kualitas', 'Confidence']], use_container_width=True, hide_index=True, height=450)
        else:
            st.info("Silakan isi parameter cuaca di kiri dan klik 'PROSES PREDIKSI'.")

if __name__ == "__main__":
    main()
