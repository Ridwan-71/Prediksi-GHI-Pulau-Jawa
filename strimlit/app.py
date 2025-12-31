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
# CUSTOM CSS (ADAPTIF & MODERN)
# ========================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #FF8C00 0%, #FFD700 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: #1e1e1e !important;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }

    .announce-box {
        color: var(--text-color); 
        background-color: rgba(255, 140, 0, 0.05);
        border: 1px solid rgba(255, 140, 0, 0.3);
        border-left: 8px solid #FF8C00;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 30px;
        line-height: 1.7;
    }
    
    .source-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 15px;
    }

    .source-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px dashed #FF8C00;
    }

    .announce-link {
        color: #FF4500 !important; 
        text-decoration: underline;
        font-weight: 700;
    }

    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(135deg, #FF8C00 0%, #FFA500 100%);
        color: white !important;
        font-weight: bold;
        height: 3.5rem;
        border: none;
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
            return pd.DataFrame({'hour': range(24), 'GHI': ghi_values}), True, f"Profil {target_col} Aktif"
    except Exception as e:
        return None, False, f"Error: {str(e)}"
    return None, False, "Format tidak sesuai"

def get_status(ghi):
    if ghi == 0: return "🌙 Malam Hari"
    if ghi < 200: return "🔴 Intensitas Rendah"
    if ghi <= 600: return "🟡 Intensitas Sedang"
    return "🟢 Intensitas Tinggi (Optimal)"

def run_prediction(historical_data, temp, hum, press):
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
    # Header dengan Ikon Matahari
    st.markdown("""
        <div class="main-header">
            <h1 style='margin:0;'>🌞 PREDIKSI RADIASI MATAHARI (GHI)</h1>
            <p style='font-size: 1.2rem; font-weight: 600;'>Dashboard Monitoring Real-Time & Estimasi 24 Jam</p>
        </div>
    """, unsafe_allow_html=True)

    # Box Panduan
    st.markdown("""
        <div class="announce-box">
            <b>📋 Panduan Persiapan Data & Parameter:</b>
            Untuk mendapatkan akurasi estimasi yang maksimal, pastikan Anda melengkapi data berikut:
            <div class="source-grid">
                <div class="source-item">
                    <b>1. Profil Historis GHI</b><br>
                    Gunakan file <b>Average hourly profiles (.xlsx)</b> dari 
                    <a href="https://globalsolaratlas.info/" target="_blank" class="announce-link">Global Solar Atlas</a>.
                    Upload file pada panel sebelah kiri.
                </div>
                <div class="source-item">
                    <b>2. Data Cuaca Real-Time</b><br>
                    Cek data <b>Suhu, Kelembapan, dan Tekanan Udara</b> terkini melalui 
                    <a href="https://www.google.com/search?q=cuaca+hari+ini" target="_blank" class="announce-link">Google Search</a> 
                    sesuai lokasi Anda.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("📂 Pengaturan Data")
        uploaded_file = st.file_uploader("Upload Profil Solar Atlas", type=['xlsx'])
        if uploaded_file:
            data, success, msg = load_excel_data(uploaded_file)
            if success:
                st.session_state['hist_data'] = data
                st.success(msg)
        
        if 'hist_data' not in st.session_state:
            st.session_state['hist_data'] = pd.DataFrame({'hour': range(24), 'GHI': [0]*24})
        
        st.markdown("---")
        st.info("Sistem ini memproyeksikan radiasi berdasarkan kombinasi data historis dan kondisi cuaca saat ini.")

    col_in, col_out = st.columns([1, 2], gap="large")

    with col_in:
        st.subheader("📝 Input Parameter Cuaca")
        with st.form("input_form"):
            t = st.slider("Temperatur Udara (°C)", 15, 45, 30)
            h = st.slider("Kelembapan Udara (%)", 0, 100, 60)
            p = st.number_input("Tekanan Udara (hPa)", 900, 1100, 1010)
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("MULAI PROSES PREDIKSI")

        if submit:
            st.session_state['res'] = run_prediction(st.session_state['hist_data'], t, h, p)

    with col_out:
        if 'res' in st.session_state:
            df_res = st.session_state['res']
            now_row = df_res.iloc[0]
            next_row = df_res.iloc[1]

            # Dashboard Metrics
            m1, m2, m3 = st.columns(3)
            with m1: st.metric(f"Jam Sekarang ({now_row['Jam']})", f"{now_row['GHI (W/m²)']} W/m²")
            with m2: 
                diff = int(next_row['GHI (W/m²)'] - now_row['GHI (W/m²)'])
                st.metric(f"Prediksi 1 Jam Ke Depan", f"{next_row['GHI (W/m²)']} W/m²", delta=f"{diff} W/m²")
            with m3: st.metric("Status Radiasi", now_row['Kualitas'].split(' ')[1])

            # Chart Proyeksi 24 Jam
            st.markdown("---")
            st.subheader("📈 Proyeksi Radiasi 24 Jam Ke Depan")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_res['Waktu'], y=df_res['GHI (W/m²)'],
                mode='lines+markers', line=dict(color='#FF8C00', width=4),
                fill='tozeroy', fillcolor='rgba(255, 140, 0, 0.1)', name='Estimasi GHI'
            ))
            fig.update_layout(
                margin=dict(l=0,r=0,t=20,b=0), height=380,
                hovermode="x unified",
                template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detail Table
            st.subheader("📋 Rincian Estimasi Per Jam")
            st.dataframe(df_res[['Jam', 'GHI (W/m²)', 'Kualitas', 'Confidence']], use_container_width=True, hide_index=True, height=450)
        else:
            st.warning("⚠️ Silakan lengkapi parameter cuaca di panel kiri, lalu tekan tombol prediksi.")

if __name__ == "__main__":
    main()
