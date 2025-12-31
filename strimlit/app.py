def load_excel_data(uploaded_file):
    """Load data GHI dari file Excel dengan pembersihan otomatis"""
    try:
        # Membaca file excel
        df = pd.read_excel(uploaded_file)
        
        # 1. Mencari kolom Waktu (mencari kata 'time', 'date', atau 'waktu')
        timestamp_col = None
        for col in df.columns:
            if any(key in col.lower() for key in ['time', 'date', 'waktu', 'tanggal']):
                timestamp_col = col
                break
        
        # 2. Mencari kolom GHI (mencari kata 'ghi', 'irradiance', atau 'radiasi')
        ghi_col = None
        for col in df.columns:
            if any(key in col.lower() for key in ['ghi', 'irradiance', 'radiasi', 'watt']):
                ghi_col = col
                break
        
        if timestamp_col and ghi_col:
            # Mengubah kolom waktu menjadi format datetime yang benar
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Membersihkan data dari nilai kosong (NaN)
            df = df.dropna(subset=[timestamp_col, ghi_col])
            
            # Mengambil hanya kolom yang diperlukan
            result = df[[timestamp_col, ghi_col]].copy()
            result.columns = ['timestamp', 'GHI']
            
            # Mengurutkan berdasarkan waktu (penting untuk ARIMA)
            result = result.sort_values('timestamp')
            
            return result, True, f"✅ Berhasil memuat {len(result)} data point dari kolom '{timestamp_col}' dan '{ghi_col}'"
        else:
            return None, False, "❌ Format kolom tidak dikenali. Pastikan ada kolom bernama 'Waktu' dan 'GHI'"
            
    except Exception as e:
        return None, False, f"❌ Terjadi kesalahan: {str(e)}"
