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
    page_title="GHI Predictor Pro | Jawa Series",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================
# CUSTOM CSS (PERBAIKAN VISUAL & KONTRAS)
# ========================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header Utama */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* FIX: Announcement Box (Kontras Tinggi) */
    .announce-box {
        background-color: #f0f7ff; /* Latar biru sangat muda */
        border-left: 6px solid #667eea;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        line-height: 1.8; /* Memberi ruang antar baris agar tidak menimpa */
    }
    
    .announce-box strong {
        color: #1e3a8a; /* Biru tua gelap */
        font-size: 1.1rem;
        display: block;
        margin-bottom: 5px;
    }

    .announce-box p {
        color: #2d3748; /* Abu-abu sangat gelap */
        margin: 0;
        font-size: 1rem;
    }

    /* Link Magenta Terang agar terlihat jelas */
    .announce-link {
        color: #e21dca !important; 
        text-decoration: underline;
        font-weight: 700;
    }

    /* Button Styling */
    .stButton>button {
        border-radius: 10px;
        height: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
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
    return None, False, "Format file tidak dikenali"

def get_status(ghi):
    if ghi == 0: return "🌙 Malam"
    if ghi < 200: return "🔴 Buruk"
    if ghi <= 600: return "🟡 Cukup Baik"
    return "🟢 Baik"

def run_prediction(historical_data, temp, hum, press, model):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    base_dict = historical_data.set_index('hour')['GHI'].to_dict()
    # Faktor koreksi lingkungan
    w_factor = (1 + (temp - 25) * 0.003) * (1 - (hum / 100) * 0.15) * (press / 1013)
    
    results = []
    for i in range(25):
        future = now + timedelta(hours=i)
        hr = future.hour
        val = base_dict.get(hr, 0) * w_factor
        if hr < 6 or hr >= 18: val = 0 # Proteksi malam hari
        if model in ['arima', 'sarima']: val *= (1 + np.random.randn() * 0.03)
        
        final_ghi = round(max(0, val))
        results.append({
            "Waktu": future,
            "Jam": future.strftime('%H:%M'),
            "GHI (W/m²)": final_ghi,
            "Kualitas": get_status(final_ghi),
            "Confidence": f"{max(60, 95 - i*1.4):.1f}%"
        })
    return pd.DataFrame(results)

# ========================================================================
# APLIKASI UTAMA
# ========================================================================
def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1 style='color: white; margin:0;'>SISTEM PREDIKSI GHI REAL-TIME</h1>
            <p style='color: #e2e8f0; font-size: 1.1rem;'>Monitoring Radiasi Matahari Pulau Jawa</p>
        </div>
    """, unsafe_allow_html=True)

    # Perbaikan Announcement Box
    st.markdown("""
        <div class="announce-box">
            <strong>📢 INFORMASI SUMBER DATA</strong>
            <p>
                Prediksi ini menggunakan profil historis dari 
                <a href="https://globalsolaratlas.info/" target="_blank" class="announce-link">Global Solar Atlas</a>. 
                Sistem secara otomatis menyesuaikan data dengan waktu lokal Anda saat ini untuk memberikan estimasi 24 jam ke depan.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📂 Input Data Profil")
        uploaded_file = st.file_uploader("Upload Excel Solar Atlas", type=['xlsx'])
        if uploaded_file:
            data, success, msg = load_excel_data(uploaded_file)
            if success:
                st.session_state['hist_data'] = data
                st.success(msg)
        
        if 'hist_data' not in st.session_state:
            st.session_state['hist_data'] = pd.DataFrame({'hour': range(24), 'GHI': [0]*24})
        
        st.markdown("---")
        model_type = st.selectbox("Algoritma", ['ARIMA', 'SARIMA', 'Exponential'])

    # Input & Output
    col_in, col_out = st.columns([1, 2], gap="large")

    with col_in:
        st.subheader("🌦️ Kondisi Lingkungan")
        with st.form("input_form"):
            t = st.slider("Temperatur (°C)", 15, 45, 30)
            h = st.slider("Kelembapan (%)", 0, 100, 60)
            p = st.number_input("Tekanan (hPa)", 900, 1100, 1010)
            st.markdown("---")
            submit = st.form_submit_button("HITUNG PREDIKSI")

        if submit:
            st.session_state['res'] = run_prediction(st.session_state['hist_data'], t, h, p, model_type.lower())

    with col_out:
        if 'res' in st.session_state:
            df_res = st.session_state['res']
            now_row = df_res.iloc[0]
            next_row = df_res.iloc[1]

            # Metric Cards
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Saat Ini ({now_row['Jam']})", f"{now_row['GHI (W/m²)']}", now_row['Kualitas'])
            
            diff = int(next_row['GHI (W/m²)'] - now_row['GHI (W/m²)'])
            m2.metric(f"Jam {next_row['Jam']}", f"{next_row['GHI (W/m²)']}", f"{diff} W/m²")
            
            m3.metric("Puncak Hari Ini", f"{df_res['GHI (W/m²)'].max()} W/m²")

            # Chart
            st.markdown("---")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_res['Waktu'], y=df_res['GHI (W/m²)'],
                mode='lines+markers',
                line=dict(color='#667eea', width=4),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)',
                name='GHI'
            ))
            fig.update_layout(
                xaxis_title="Waktu (24 Jam)",
                yaxis_title="GHI (W/m²)",
                height=350,
                margin=dict(l=0,r=0,t=10,b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detail Table
            st.subheader("📋 Detail Estimasi 24 Jam")
            st.dataframe(
                df_res[['Jam', 'GHI (W/m²)', 'Kualitas', 'Confidence']],
                use_container_width=True,
                hide_index=True,
                height=500
            )
        else:
            st.info("Atur parameter cuaca di kiri dan klik 'HITUNG PREDIKSI' untuk melihat dashboard.")

if __name__ == "__main__":
    main()
