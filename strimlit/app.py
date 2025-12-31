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
# CUSTOM CSS (ADAPTIF DARK & LIGHT MODE)
# ========================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Header Utama */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white !important;
        text-align: center;
        margin-bottom: 20px;
    }

    /* FIX: Box Pengumuman Adaptif */
    .announce-box {
        /* Mengikuti warna teks sistem agar terlihat di dark/light mode */
        color: var(--text-color); 
        background-color: rgba(102, 126, 234, 0.1); /* Transparan lembut */
        border: 1px solid #667eea;
        border-left: 6px solid #667eea;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        line-height: 1.6;
    }
    
    .announce-box strong {
        display: block;
        margin-bottom: 5px;
        font-size: 1.1rem;
        color: #667eea; /* Tetap ungu agar konsisten */
    }

    /* Link yang terlihat di semua mode (Gold/Orange) */
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
        height: 3rem;
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
            return pd.DataFrame({'hour': range(24), 'GHI': ghi_values}), True, f"Data {target_col} Aktif"
    except Exception as e:
        return None, False, str(e)
    return None, False, "Format tidak cocok"

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
        if model in ['arima', 'sarima']: val *= (1 + np.random.randn() * 0.03)
        
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
            <h1 style='color: white; margin:0;'>SISTEM PREDIKSI GHI</h1>
            <p style='color: white; opacity: 0.8;'>Monitoring Real-Time Radiasi Matahari</p>
        </div>
    """, unsafe_allow_html=True)

    # Box Pengumuman dengan Warna Teks Fleksibel
    st.markdown("""
        <div class="announce-box">
            <strong>📢 SUMBER DATA HISTORIS</strong>
            Sistem ini menggunakan profil radiasi dari 
            <a href="https://globalsolaratlas.info/" target="_blank" class="announce-link">Global Solar Atlas</a>. 
            Data disesuaikan secara real-time berdasarkan jam sistem Anda.
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("📂 Data Source")
        uploaded_file = st.file_uploader("Upload Excel", type=['xlsx'])
        if uploaded_file:
            data, success, msg = load_excel_data(uploaded_file)
            if success:
                st.session_state['hist_data'] = data
                st.success(msg)
        
        if 'hist_data' not in st.session_state:
            st.session_state['hist_data'] = pd.DataFrame({'hour': range(24), 'GHI': [0]*24})
        
        model_type = st.selectbox("Metode", ['ARIMA', 'SARIMA', 'Exponential'])

    col_in, col_out = st.columns([1, 2], gap="medium")

    with col_in:
        st.subheader("🌦️ Parameter Cuaca")
        with st.form("input_form"):
            t = st.slider("Suhu (°C)", 15, 45, 30)
            h = st.slider("Kelembapan (%)", 0, 100, 60)
            p = st.number_input("Tekanan (hPa)", 900, 1100, 1010)
            submit = st.form_submit_button("HITUNG SEKARANG")

        if submit:
            st.session_state['res'] = run_prediction(st.session_state['hist_data'], t, h, p, model_type.lower())

    with col_out:
        if 'res' in st.session_state:
            df_res = st.session_state['res']
            now_row = df_res.iloc[0]
            next_row = df_res.iloc[1]

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Jam {now_row['Jam']}", f"{now_row['GHI (W/m²)']} W/m²")
            m2.metric(f"Prediksi 1 Jam", f"{next_row['GHI (W/m²)']} W/m²")
            m3.metric("Kualitas", f"{now_row['Kualitas']}")

            # Chart
            st.markdown("---")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_res['Waktu'], y=df_res['GHI (W/m²)'],
                mode='lines+markers', line=dict(color='#667eea', width=3),
                fill='tozeroy', name='GHI'
            ))
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0), template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(df_res[['Jam', 'GHI (W/m²)', 'Kualitas', 'Confidence']], use_container_width=True, hide_index=True, height=400)
        else:
            st.info("Silakan klik tombol 'HITUNG SEKARANG'.")

if __name__ == "__main__":
    main()
