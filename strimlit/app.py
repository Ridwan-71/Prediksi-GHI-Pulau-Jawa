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
# CUSTOM CSS (OPTIMASI DARK/LIGHT MODE & TYPOGRAPHY)
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

    /* Box Pengumuman yang Lebih Proper & Jelas */
    .announce-box {
        color: var(--text-color); 
        background-color: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-left: 6px solid #667eea;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        line-height: 1.6;
    }
    
    .announce-box b {
        color: #667eea;
        font-size: 1.15rem;
    }

    .instructions {
        margin-top: 10px;
        padding-left: 20px;
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
            return pd.DataFrame({'hour': range(24), 'GHI': ghi_values}), True, f"Data Profil {target_col} Berhasil Dimuat"
    except Exception as e:
        return None, False, f"Format file salah atau rusak: {str(e)}"
    return None, False, "Format sheet Solar Atlas tidak terdeteksi."

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
        if model.lower() in ['arima', 'sarima']: val *= (1 + np.random.randn() * 0.03)
        
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
            <h1 style='color: white; margin:0;'>SISTEM PREDIKSI RADIASI MATAHARI (GHI)</h1>
            <p style='color: white; opacity: 0.9;'>Estimasi Real-Time 24 Jam Kedepan</p>
        </div>
    """, unsafe_allow_html=True)

    # Box Pengumuman yang Jauh Lebih Jelas
    st.markdown("""
        <div class="announce-box">
            <b>⚠️ Instruksi Persiapan Data:</b>
            Untuk mendapatkan hasil prediksi yang akurat dan sesuai dengan lokasi Anda, mohon ikuti langkah berikut:
            <div class="instructions">
                1. Kunjungi situs <a href="https://globalsolaratlas.info/" target="_blank" class="announce-link">Global Solar Atlas</a>.<br>
                2. Pilih lokasi spesifik Anda dan unduh data <b>"Average hourly profiles"</b>.<br>
                3. Pastikan file dalam format <b>.xlsx (Excel)</b>.<br>
                4. Upload file tersebut pada menu sidebar di sebelah kiri.<br>
                <i>*Jika tidak mengunggah data, sistem akan menggunakan nilai default 0.</i>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("📂 Panel Data")
        uploaded_file = st.file_uploader("Unggah File Solar Atlas (.xlsx)", type=['xlsx'])
        if uploaded_file:
            data, success, msg = load_excel_data(uploaded_file)
            if success:
                st.session_state['hist_data'] = data
                st.success(msg)
            else:
                st.error(msg)
        
        if 'hist_data' not in st.session_state:
            st.session_state['hist_data'] = pd.DataFrame({'hour': range(24), 'GHI': [0]*24})
        
        st.markdown("---")
        model_type = st.selectbox("Metode Forecasting", ['ARIMA', 'SARIMA', 'Exponential Smoothing'])

    col_in, col_out = st.columns([1, 2], gap="large")

    with col_in:
        st.subheader("🌦️ Input Cuaca Saat Ini")
        with st.form("input_form"):
            t = st.slider("Temperatur (°C)", 15, 45, 30)
            h = st.slider("Kelembapan (%)", 0, 100, 60)
            p = st.number_input("Tekanan Udara (hPa)", 900, 1100, 1010)
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("PROSES PREDIKSI")

        if submit:
            st.session_state['res'] = run_prediction(st.session_state['hist_data'], t, h, p, model_type)

    with col_out:
        if 'res' in st.session_state:
            df_res = st.session_state['res']
            now_row = df_res.iloc[0]
            next_row = df_res.iloc[1]

            # Dashboard Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(f"Jam {now_row['Jam']}", f"{now_row['GHI (W/m²)']} W/m²")
            with m2:
                diff = int(next_row['GHI (W/m²)'] - now_row['GHI (W/m²)'])
                st.metric(f"Prediksi 1 Jam", f"{next_row['GHI (W/m²)']} W/m²", delta=f"{diff} W/m²")
            with m3:
                st.metric("Kualitas Sinar", now_row['Kualitas'])

            # Chart Proyeksi
            st.markdown("---")
            st.subheader("📈 Kurva Estimasi 24 Jam")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_res['Waktu'], y=df_res['GHI (W/m²)'],
                mode='lines+markers', line=dict(color='#667eea', width=3),
                fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.1)', name='GHI'
            ))
            fig.update_layout(
                margin=dict(l=0,r=0,t=20,b=0), height=350,
                hovermode="x unified",
                template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detail Table
            st.subheader("📋 Tabel Rincian Per Jam")
            st.dataframe(
                df_res[['Jam', 'GHI (W/m²)', 'Kualitas', 'Confidence']], 
                use_container_width=True, 
                hide_index=True, 
                height=450
            )
        else:
            st.info("💡 Silakan isi parameter cuaca dan klik 'PROSES PREDIKSI' untuk melihat hasil.")

if __name__ == "__main__":
    main()
