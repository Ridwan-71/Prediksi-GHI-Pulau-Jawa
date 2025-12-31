Dashboard Prediksi Radiasi Matahari (GHI)
Berikut adalah panduan untuk menjalankan dan menggunakan aplikasi Prediksi GHI Real-Time.

MENGGUNAKAN LOCAL HOST

LANGKAH 1: Mengetahui Lokasi Folder
- Pastikan anda telah mengunduh atau mengekstrak folder projek ini.
- Buka File Manager, cari folder projek tersebut.
- Klik kanan pada folder, pilih Properties.
- Salin atau catat lokasi folder pada bagian Location (contoh: C:\Users\Nama_PC\Downloads\GHI_Project).

LANGKAH 2: Menjalankan Program melalui Python Editor (Spyder/VS Code)
- Buka aplikasi editor anda (contoh: Spyder).
- Klik Open File, cari folder projek dan buka file utama bernama app.py.
- Pastikan semua pustaka (library) seperti streamlit, pandas, numpy, dan plotly sudah terpasang.
- Klik ikon Run (segitiga hijau) atau tekan F5.

LANGKAH 3: Mengakses Aplikasi melalui Streamlit
- Biarkan program tetap berjalan.
- Buka Anaconda Prompt atau Command Prompt (CMD).
- Ketik perintah berikut: conda activate base (jika menggunakan Anaconda).
- Masuk ke lokasi folder dengan perintah: cd /d lokasi_folder_anda
- Jalankan aplikasi dengan perintah: streamlit run app.py
- Browser akan terbuka secara otomatis ke alamat http://localhost:8501.

MENGGUNAKAN WEBSITE (ONLINE)
Pastikan peranti anda terhubung ke internet.

Akses aplikasi melalui pautan rasmi (jika sudah dideploy): https://nama-aplikasi-anda.streamlit.app/

CARA MENGGUNAKAN PROGRAM
1. Menyiapkan Data Profil Historis
- Kunjungi Global Solar Atlas: https://globalsolaratlas.info/
- Cari lokasi yang diinginkan pada peta.
- Muat turun data dalam format "Average hourly profiles" (Fail .xlsx).
- Di dalam aplikasi, gunakan menu Sidebar Kiri untuk mengunggah fail Excel tersebut.

2. Mengambil Data Cuaca Real-Time
- Untuk mendapatkan hasil yang tepat, cari data cuaca semasa melalui Google Search (contoh kata kunci: "Cuaca Jakarta hari ini").
- Dapatkan nilai untuk:
- Suhu (°C)
- Kelembapan (%)
- Tekanan Udara (hPa)
- Masukkan nilai-nilai tersebut ke dalam panel Input Parameter Cuaca di aplikasi.

3. Memproses Prediksi
- Setelah data profil diunggah dan parameter cuaca diatur, klik tombol "PROSES PREDIKSI".
- Dashboard akan menampilkan:
- Metrik Real-Time: Perbandingan GHI saat ini dan prediksi 1 jam ke depan.
- Grafik Interaktif: Kurva estimasi radiasi matahari selama 24 jam ke depan.
- Tabel Rincian: Detail data per jam lengkap dengan status kualitas radiasi dan tingkat kepercayaan (confidence).

  Note:
- Aplikasi ini memproses data secara real-time berdasarkan waktu sistem komputer anda.
- Jika fail Excel tidak diunggah, sistem akan menggunakan nilai default 0.
