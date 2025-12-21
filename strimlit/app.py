"""
Sistem Prediksi GHI di Pulau Jawa
Aplikasi Streamlit untuk prediksi Global Horizontal Irradiance
Deploy ke: streamlit.app
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi GHI Pulau Jawa",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #667eea;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
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
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk generate data historis simulasi
@st.cache_data
def generate_historical_data(days=30):
    """Generate data GHI historis untuk simulasi"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=days*24,
        freq='H'
    )
    
    ghi_values = []
    for date in dates:
        hour = date.hour
        if 6 <= hour <= 18:
            sun_angle = np.sin(np.pi * (hour - 6) / 12)
            base_ghi = 1000 * sun_angle
            ghi = base_ghi * (0.7 + np.random.random() * 0.3)
        else:
            ghi = 0
        ghi_values.append(max(0, ghi))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'GHI': ghi_values
    })
    return df

# Fungsi untuk load data dari Excel
def load_excel_data(uploaded_file):
    """Load data GHI dari file Excel"""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Cari kolom timestamp
        timestamp_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                timestamp_col = col
                break
        
        # Cari kolom GHI
        ghi_col = None
        for col in df.columns:
            if 'ghi' in col.lower() or 'irradiance' in col.lower():
                ghi_col = col
                break
        
        if timestamp_col and ghi_col:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            result = df[[timestamp_col, ghi_col]].copy()
            result.columns = ['timestamp', 'GHI']
            return result, True, f"✅ Berhasil memuat {len(result)} data point"
        else:
            return None, False, "❌ Kolom timestamp atau GHI tidak ditemukan"
            
    except Exception as e:
        return None, False, f"❌ Error: {str(e)}"

# Fungsi prediksi ARIMA sederhana
def predict_arima(historical_data, steps=24):
    """Prediksi menggunakan metode ARIMA sederhana"""
    # Ambil data 168 jam terakhir (7 hari)
    recent_data = historical_data['GHI'].tail(168).values
    
    # Hitung seasonal pattern (24 jam)
    seasonal_pattern = np.zeros(24)
    seasonal_count = np.zeros(24)
    
    for i, value in enumerate(recent_data):
        hour = i % 24
        seasonal_pattern[hour] += value
        seasonal_count[hour] += 1
    
    seasonal_pattern = seasonal_pattern / np.maximum(seasonal_count, 1)
    
    predictions = []
    current_hour = datetime.now().hour
    
    for i in range(steps):
        hour = (current_hour + i) % 24
        base_pred = seasonal_pattern[hour]
        
        # AR component
        if i > 0:
            ar_component = predictions[-1] * 0.3
            base_pred = base_pred * 0.7 + ar_component
        
        # Tambah noise
        if base_pred > 0:
            base_pred *= (1 + np.random.randn() * 0.05)
        
        predictions.append(max(0, base_pred))
    
    return predictions

# Fungsi prediksi SARIMA
def predict_sarima(historical_data, steps=24):
    """Prediksi menggunakan metode SARIMA"""
    recent_data = historical_data['GHI'].tail(168).values
    
    seasonal_pattern = np.zeros(24)
    seasonal_count = np.zeros(24)
    
    for i, value in enumerate(recent_data):
        hour = i % 24
        seasonal_pattern[hour] += value
        seasonal_count[hour] += 1
    
    seasonal_pattern = seasonal_pattern / np.maximum(seasonal_count, 1)
    
    predictions = []
    current_hour = datetime.now().hour
    
    for i in range(steps):
        hour = (current_hour + i) % 24
        base_pred = seasonal_pattern[hour]
        
        # Seasonal AR component
        if i > 0:
            ar_component = predictions[-1] * 0.25
            seasonal_component = predictions[max(0, i-12)] * 0.15 if i >= 12 else 0
            base_pred = base_pred * 0.6 + ar_component + seasonal_component
        
        if base_pred > 0:
            base_pred *= (1 + np.random.randn() * 0.05)
        
        predictions.append(max(0, base_pred))
    
    return predictions

# Fungsi prediksi Exponential Smoothing
def predict_exponential(historical_data, steps=24):
    """Prediksi menggunakan Exponential Smoothing"""
    recent_data = historical_data['GHI'].tail(168).values
    
    seasonal_pattern = np.zeros(24)
    seasonal_count = np.zeros(24)
    
    for i, value in enumerate(recent_data):
        hour = i % 24
        seasonal_pattern[hour] += value
        seasonal_count[hour] += 1
    
    seasonal_pattern = seasonal_pattern / np.maximum(seasonal_count, 1)
    
    predictions = []
    current_hour = datetime.now().hour
    alpha = 0.3
    
    for i in range(steps):
        hour = (current_hour + i) % 24
        base_pred = seasonal_pattern[hour]
        
        if i > 0:
            smoothed = alpha * base_pred + (1 - alpha) * predictions[-1]
            base_pred = smoothed
        
        if base_pred > 0:
            base_pred *= (1 + np.random.randn() * 0.05)
        
        predictions.append(max(0, base_pred))
    
    return predictions

# Fungsi prediksi Moving Average
def predict_moving_average(historical_data, steps=24):
    """Prediksi menggunakan Moving Average"""
    recent_data = historical_data['GHI'].tail(168).values
    
    seasonal_pattern = np.zeros(24)
    seasonal_count = np.zeros(24)
    
    for i, value in enumerate(recent_data):
        hour = i % 24
        seasonal_pattern[hour] += value
        seasonal_count[hour] += 1
    
    seasonal_pattern = seasonal_pattern / np.maximum(seasonal_count, 1)
    
    predictions = []
    current_hour = datetime.now().hour
    
    for i in range(steps):
        hour = (current_hour + i) % 24
        base_pred = seasonal_pattern[hour]
        
        if i >= 2:
            ma3 = (predictions[-1] + predictions[-2] + base_pred) / 3
            base_pred = ma3
        
        if base_pred > 0:
            base_pred *= (1 + np.random.randn() * 0.05)
        
        predictions.append(max(0, base_pred))
    
    return predictions

# Fungsi prediksi dengan faktor cuaca
def predict_with_weather(historical_data, temp, humidity, pressure, model_type='arima', steps=24):
    """Prediksi GHI dengan mempertimbangkan cuaca"""
    
    # Pilih model
    if model_type == 'arima':
        base_predictions = predict_arima(historical_data, steps)
    elif model_type == 'sarima':
        base_predictions = predict_sarima(historical_data, steps)
    elif model_type == 'exponential':
        base_predictions = predict_exponential(historical_data, steps)
    else:
        base_predictions = predict_moving_average(historical_data, steps)
    
    # Faktor cuaca
    temp_factor = 1 + (temp - 25) * 0.003
    humidity_factor = 1 - (humidity / 100) * 0.15
    pressure_factor = (pressure - 1013) / 1013 * 0.05 + 1
    
    # Aplikasikan faktor cuaca dengan decay
    adjusted_predictions = []
    for i, pred in enumerate(base_predictions):
        weather_impact = np.exp(-i / 12)
        weather_factor = (
            1 + 
            (temp_factor - 1) * weather_impact +
            (humidity_factor - 1) * weather_impact +
            (pressure_factor - 1) * weather_impact
        )
        adjusted_pred = pred * weather_factor
        adjusted_predictions.append(max(0, min(1200, adjusted_pred)))
    
    # Buat DataFrame hasil
    current_time = datetime.now()
    future_times = [current_time + timedelta(hours=i) for i in range(steps)]
    
    results = pd.DataFrame({
        'Waktu': future_times,
        'Jam': [t.strftime('%H:%M') for t in future_times],
        'GHI': [round(p) for p in adjusted_predictions],
        'Confidence': [max(60, 95 - i*1.5) for i in range(steps)]
    })
    
    return results

# Main App
def main():
    # Header
    st.markdown("<h1>🌞 Sistem Prediksi GHI di Pulau Jawa</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1em;'>Prediksi Global Horizontal Irradiance untuk 24 Jam Ke Depan</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        # Upload data
        st.subheader("📁 Data Historis")
        uploaded_file = st.file_uploader(
            "Upload File Excel (Opsional)",
            type=['xlsx', 'xls'],
            help="Upload data GHI historis untuk prediksi lebih akurat"
        )
        
        if uploaded_file:
            data, success, message = load_excel_data(uploaded_file)
            if success:
                st.success(message)
                st.session_state['historical_data'] = data
            else:
                st.error(message)
                if 'historical_data' not in st.session_state:
                    st.session_state['historical_data'] = generate_historical_data()
        else:
            if 'historical_data' not in st.session_state:
                st.session_state['historical_data'] = generate_historical_data()
                st.info("💡 Menggunakan data simulasi")
        
        st.markdown("---")
        
        # Model selection
        st.subheader("📊 Model Prediksi")
        model_type = st.selectbox(
            "Pilih Model",
            ['arima', 'sarima', 'exponential', 'moving_average'],
            format_func=lambda x: {
                'arima': 'ARIMA',
                'sarima': 'SARIMA (Seasonal)',
                'exponential': 'Exponential Smoothing',
                'moving_average': 'Moving Average'
            }[x]
        )
        
        st.markdown("---")
        
        # Info
        st.subheader("💡 Informasi Model")
        st.markdown("""
        - **ARIMA**: Model klasik time series
        - **SARIMA**: Dengan pola musiman
        - **Exp. Smoothing**: Pembobotan waktu
        - **Moving Average**: Rata-rata bergerak
        """)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Input Data Cuaca")
        
        with st.form("weather_form"):
            temperature = st.number_input(
                "🌡️ Suhu (°C)",
                min_value=15.0,
                max_value=40.0,
                value=28.0,
                step=0.1,
                help="Range normal: 20-35°C"
            )
            
            humidity = st.number_input(
                "💧 Kelembapan (%)",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=0.1,
                help="Range: 0-100%"
            )
            
            pressure = st.number_input(
                "📊 Tekanan Udara (hPa)",
                min_value=950.0,
                max_value=1050.0,
                value=1013.0,
                step=0.1,
                help="Range normal: 980-1050 hPa"
            )
            
            submitted = st.form_submit_button("🔮 Prediksi 24 Jam", use_container_width=True)
        
        if submitted:
            with st.spinner("🔄 Memproses prediksi..."):
                # Lakukan prediksi
                predictions = predict_with_weather(
                    st.session_state['historical_data'],
                    temperature,
                    humidity,
                    pressure,
                    model_type
                )
                
                st.session_state['predictions'] = predictions
                st.session_state['model_name'] = model_type
                st.success("✅ Prediksi berhasil!")
    
    with col2:
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            
            # Statistics
            st.subheader("📊 Statistik Prediksi")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Max GHI", f"{predictions['GHI'].max()} W/m²")
            
            with col_stat2:
                st.metric("Rata-rata", f"{predictions['GHI'].mean():.0f} W/m²")
            
            with col_stat3:
                st.metric("Total Energi", f"{predictions['GHI'].sum()/1000:.1f} kWh/m²")
            
            with col_stat4:
                model_names = {
                    'arima': 'ARIMA',
                    'sarima': 'SARIMA',
                    'exponential': 'Exp. SM',
                    'moving_average': 'MA'
                }
                st.metric("Model", model_names.get(st.session_state.get('model_name', 'arima')))
            
            st.markdown("---")
            
            # Chart
            st.subheader("📈 Grafik Prediksi GHI")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=predictions['Jam'],
                y=predictions['GHI'],
                mode='lines+markers',
                name='GHI Prediksi',
                line=dict(color='#667eea', width=3),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            
            fig.update_layout(
                xaxis_title="Waktu (Jam)",
                yaxis_title="GHI (W/m²)",
                hovermode='x unified',
                height=400,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Table
            st.subheader("📋 Tabel Prediksi Detail")
            
            # Tambahkan kolom status
            def get_status(ghi):
                if ghi > 700:
                    return "☀️ Optimal"
                elif ghi > 300:
                    return "⛅ Baik"
                elif ghi > 0:
                    return "🌤️ Rendah"
                else:
                    return "🌙 Malam"
            
            display_df = predictions[['Jam', 'GHI', 'Confidence']].copy()
            display_df['Status'] = predictions['GHI'].apply(get_status)
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1f}%")
            display_df['GHI'] = display_df['GHI'].apply(lambda x: f"{x} W/m²")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = predictions.to_csv(index=False)
            st.download_button(
                label="📥 Download Hasil (CSV)",
                data=csv,
                file_name=f"prediksi_ghi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.info("👈 Masukkan data cuaca di sebelah kiri dan klik 'Prediksi 24 Jam'")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p><strong>💡 Catatan:</strong> Akurasi prediksi meningkat dengan data historis yang lebih lengkap dan berkualitas.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()