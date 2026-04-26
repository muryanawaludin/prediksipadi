import streamlit as st
import pandas as pd
import joblib

# Load artifacts
rf = joblib.load('random_forest_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_columns = joblib.load('feature_columns.pkl')
categorical_cols = joblib.load('categorical_cols.pkl')
categorical_unique_values = joblib.load('categorical_unique_values.pkl')

st.set_page_config(page_title="Prediksi Varietas Padi", layout="wide")
st.title("🌾 Prediksi Varietas Padi")
st.markdown("Masukkan data sesuai dengan karakteristik berikut:")

# Helper untuk membuat input berdasarkan tipe kolom
def create_input_row(col_name, col_type):
    if col_type == 'categorical':
        # Pilihan dropdown
        options = categorical_unique_values[col_name]
        return st.selectbox(f"{col_name}", options, key=col_name)
    else:
        # Input angka
        return st.number_input(f"{col_name}", value=0.0, step=0.1, format="%.2f", key=col_name)

# Tentukan kolom mana saja yang numerik dan mana kategorikal
# Kita tahu kolom numerik adalah semua kolom asli kecuali categorical_cols
# Tapi karena kita menyimpan feature_columns (setelah encoding), kita perlu mengelompokkan input berdasarkan nama asli.
# Cara mudah: kita buat input untuk setiap kolom asli (sebelum encoding).
# Kolom asli yang tidak ada di categorical_cols adalah numerik.

# Dapatkan semua nama kolom asli (dari data training asli - kita simpan saat training? Kita bisa baca dari categorical_unique_values dan menambahkan kolom numerik)
# Alternatif: baca langsung dari file CSV asli? Tidak praktis. Kita akan buat daftar kolom asli secara manual berdasarkan data contoh.
# Karena data contoh sudah diberikan, kita akan daftarkan semua kolom asli (setelah drop Hectares & Agriblock).

# Dari contoh data, kolom asli (setelah drop Hectares, Agriblock) adalah:
all_original_columns = [
    'Variety',  # target, tidak dimasukkan
    'Soil Types', 'Seedrate', 'LP_Mainfield', 'Nursery', 'Nursery area (Cents)', 'LP_nurseryarea',
    'DAP_20days', 'Weed28D_thiobencarb', 'Urea_40Days', 'Potassh_50Days', 'Micronutrients_70Days',
    'Pest_60Day(in ml)', '30DRain', '30DAI', '30_50DRain', '30_50DAI', '51_70DRain', '51_70AI',
    '71_105DRain', '71_105DAI', 'Min temp_D1_D30', 'Max temp_D1_D30', 'Min temp_D31_D60',
    'Max temp_D31_D60', 'Min temp_D61_D90', 'Max temp_D61_D90', 'Min temp_D91_D120', 'Max temp_D91_D120',
    'Inst Wind Speed_D1_D30', 'Inst Wind Speed_D31_D60', 'Inst Wind Speed_D61_D90', 'Inst Wind Speed_D91_D120',
    'Wind Direction_D1_D30', 'Wind Direction_D31_D60', 'Wind Direction_D61_D90', 'Wind Direction_D91_D120',
    'Relative Humidity_D1_D30', 'Relative Humidity_D31_D60', 'Relative Humidity_D61_D90', 'Relative Humidity_D91_D120',
    'Trash(in bundles)', 'Paddy yield(in Kg)'  # Paddy yield adalah target lain? Tidak, itu hasil panen, tapi model tidak menggunakan? Di data ada kolom Paddy yield? Dari contoh data ada kolom 'Paddy yield(in Kg)' sepertinya itu target lain? Tapi kita hanya prediksi Variety. Kita abaikan saja.
]

# Namun lebih aman: ambil dari categorical_cols dan tebak numerik dari feature_columns.
# Kita akan buat daftar kolom asli yang digunakan dalam model (termasuk numerik dan kategorikal).
# Kita bisa ekstrak dari categorical_cols dan dari fitur numerik (nama kolom yang tidak di-encode).
# Karena feature_columns berisi nama setelah encoding (misal 'Soil Types_alluvial', 'Soil Types_clay', dll).
# Kita bisa mengambil prefix dari feature_columns yang merupakan kolom asli.

# Cara praktis: kita buat input untuk setiap kolom asli yang ada di categorical_cols (dropdown) dan untuk setiap kolom numerik (dari nama asli yang tidak termasuk categorical_cols).
# Kita tidak perlu input untuk kolom 'Variety' dan 'Paddy yield'.

# Daftar kolom asli yang kita perlukan (berdasarkan contoh data, tanpa Hectares, Agriblock, tanpa target Variety, tanpa Paddy yield jika tidak dipakai)
input_columns = [
    'Soil Types', 'Seedrate', 'LP_Mainfield', 'Nursery', 'Nursery area (Cents)', 'LP_nurseryarea',
    'DAP_20days', 'Weed28D_thiobencarb', 'Urea_40Days', 'Potassh_50Days', 'Micronutrients_70Days',
    'Pest_60Day(in ml)', '30DRain', '30DAI', '30_50DRain', '30_50DAI', '51_70DRain', '51_70AI',
    '71_105DRain', '71_105DAI', 'Min temp_D1_D30', 'Max temp_D1_D30', 'Min temp_D31_D60',
    'Max temp_D31_D60', 'Min temp_D61_D90', 'Max temp_D61_D90', 'Min temp_D91_D120', 'Max temp_D91_D120',
    'Inst Wind Speed_D1_D30', 'Inst Wind Speed_D31_D60', 'Inst Wind Speed_D61_D90', 'Inst Wind Speed_D91_D120',
    'Wind Direction_D1_D30', 'Wind Direction_D31_D60', 'Wind Direction_D61_D90', 'Wind Direction_D91_D120',
    'Relative Humidity_D1_D30', 'Relative Humidity_D31_D60', 'Relative Humidity_D61_D90', 'Relative Humidity_D91_D120',
    'Trash(in bundles)'
]

# Buat form
with st.form("prediction_form"):
    st.subheader("🔧 Masukkan Parameter")
    col1, col2 = st.columns(2)
    user_input = {}
    
    for i, col_name in enumerate(input_columns):
        if col_name in categorical_cols:
            # Tampilkan dropdown
            options = categorical_unique_values[col_name]
            value = st.selectbox(f"{col_name}", options, key=col_name)
        else:
            # Tampilkan input angka
            value = st.number_input(f"{col_name}", value=0.0, step=1.0, format="%.2f", key=col_name)
        user_input[col_name] = value
    
    submitted = st.form_submit_button("🌱 Prediksi Varietas")

if submitted:
    # Konversi input user ke DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Lakukan one-hot encoding seperti saat training
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)
    
    # Tambahkan kolom yang hilang (misal kategori yang tidak muncul karena user memilih salah satu)
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Urutkan kolom sesuai training
    input_encoded = input_encoded[feature_columns]
    
    # Prediksi
    pred_encoded = rf.predict(input_encoded)[0]
    pred_variety = label_encoder.inverse_transform([pred_encoded])[0]
    
    st.success(f"✨ Hasil Prediksi: **{pred_variety}**")
    
    # Optional: Tampilkan probabilitas
    proba = rf.predict_proba(input_encoded)[0]
    proba_df = pd.DataFrame({
        'Varietas': label_encoder.classes_,
        'Probabilitas': proba
    }).sort_values('Probabilitas', ascending=False)
    st.subheader("📊 Tingkat Keyakinan")
    st.dataframe(proba_df, use_container_width=True)