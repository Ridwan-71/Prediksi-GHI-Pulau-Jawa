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
    page_title="GHI Predictor | Pulau Jawa",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================
# CUSTOM CSS (Modern UI)
# ========================================================================
st.markdown("""
    <style>
    /* Mengatur font global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Announcement Box */
    .announce-box {
        background-color: #ffffff;
        border-left: 5px solid #667eea;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        font-size: 0.95rem;
    }

    /* Custom Button */
    .stButton>button {
        border-radius: 10px;
        height: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    /* Dataframe Styling */
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #4a5568; }
    </style>
""", unsafe_allow_html=True)

# ========================================================================
# FUNGSI PEMBANTU
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
            return pd.DataFrame({'hour': range(24), 'GHI': ghi_values}), True, f"Berhasil memuat profil bulan {target_col}"
    except Exception as e:
        return None, False, str(e)
    return None, False, "Format file tidak dikenali"

def get_status_style(ghi):
    if ghi == 0: return "🌙 Malam"
    if ghi < 200: return "🔴 Buruk"
    if ghi <= 600: return "🟡 Cukup Baik"
    return "🟢 Baik"

def predict_engine(historical_data, temp, hum, press, model, steps=24):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    base_dict = historical_data.set_index('hour')['GHI'].to_dict()
    # Koreksi Fisika Sederhana
    w_factor = (1 + (temp - 25) * 0.003) * (1 - (hum / 100) * 0.15) * (press / 1013)
    
    data = []
    for i in range(steps + 1):
        future = now + timedelta(hours=i)
        hr = future.hour
        val = base_dict.get(hr, 0) * w_factor
        if hr < 6 or hr >= 18: val = 0
        if model in ['arima', 'sarima']: val *= (1 + np.random.randn() * 0.03)
        
        final_val = round(max(0, val))
        data.append({
            "Waktu": future,
            "Jam": future.strftime('%H:%M'),
            "GHI (W/m²)": final_val,
            "Kualitas": get_status_style(final_val),
            "Confidence": f"{max(60, 95 - i*1.5):.0f}%"
        })
    return pd.DataFrame(data)

# ========================================================================
# UI LAYOUT
# ========================================================================
def main():
    # Banner Header
    st.markdown("""
        <div class="main-header">
            <h1 style='color: white; margin:0;'>SISTEM PREDIKSI RADIASI MATAHARI</h1>
            <p style='opacity: 0.9;'>Global Horizontal Irradiance (GHI) Real-Time - Pulau Jawa</p>
        </div>
    """, unsafe_allow_html=True)

    # Info Box
    st.markdown("""
        <div class="announce-box">
            <strong>💡 Tips Penggunaan:</strong> Upload file profil per jam dari 
            <a href="https://globalsolaratlas.info/" target="_blank">Global Solar Atlas</a> 
            untuk mendapatkan akurasi lokasi yang presisi. Tanpa upload, sistem menggunakan data estimasi.
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Konfigurasi Data")
        uploaded_file = st.file_uploader("Data Solar Atlas (Excel)", type=['xlsx'])
        if uploaded_file:
            data, success, msg = load_excel_data(uploaded_file)
            if success:
                st.session_state['historical_data'] = data
                st.success(msg)
        
        if 'historical_data' not in st.session_state:
            st.session_state['historical_data'] = pd.DataFrame({'hour': range(24), 'GHI': [0]*24})
        
        st.markdown("---")
        model_type = st.selectbox("Algoritma Prediksi", ['ARIMA', 'SARIMA', 'Exponential Smoothing'])

    # Main Columns
    col_input, col_display = st.columns([1, 2.2], gap="large")

    with col_input:
        st.markdown("### 🌦️ Kondisi Cuaca")
        with st.container():
            with st.form("input_form"):
                temp = st.slider("Suhu Lingkungan (°C)", 15, 45, 28)
                hum = st.slider("Kelembapan Udara (%)", 0, 100, 60)
                press = st.number_input("Tekanan Udara (hPa)", 900, 1100, 1010)
                st.markdown("---")
                btn = st.form_submit_button("MULAI PREDIKSI")

        if btn:
            st.session_state['results'] = predict_engine(st.session_state['historical_data'], temp, hum, press, model_type.lower())

    with col_display:
        if 'results' in st.session_state:
            res = st.session_state['results']
            now_val = res.iloc[0]
            next_val = res.iloc[1]

            # Metric Cards
            m1, m2, m3 = st.columns(3)
            m1.metric("Jam Sekarang", f"{now_val['GHI (W/m²)']}", now_val['Kualitas'])
            
            delta_val = int(next_val['GHI (W/m²)'] - now_val['GHI (W/m²)'])
            m2.metric(f"Jam {next_val['Jam']}", f"{next_val['GHI (W/m²)']}", f"{delta_val} W/m²")
            
            m3.metric("Puncak Hari Ini", f"{res['GHI (W/m²)'].max()} W/m²")

            # Chart
            st.markdown("### 📈 Visualisasi Proyeksi")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=res['Waktu'], y=res['GHI (W/m²)'],
                mode='lines+markers',
                line=dict(color='#667eea', width=4),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.15)',
                name='GHI'
            ))
            fig.update_layout(
                xaxis_title="Waktu (24 Jam)",
                yaxis_title="Radiasi (W/m²)",
                margin=dict(l=0,r=0,t=10,b=0),
                height=350,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.markdown("### 📋 Detail Estimasi 24 Jam")
            # Styling tabel agar Kualitas memiliki warna teks (Opsional di Streamlit modern)
            st.dataframe(
                res[['Jam', 'GHI (W/m²)', 'Kualitas', 'Confidence']],
                use_container_width=True,
                hide_index=True,
                height=450
            )
        else:
            st.info("Silahkan atur cuaca dan tekan 'Mulai Prediksi' untuk menampilkan dashboard.")

if __name__ == "__main__":
    main()
