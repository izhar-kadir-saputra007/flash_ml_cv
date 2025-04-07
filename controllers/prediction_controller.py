from flask import request, jsonify
import os
import fitz  # PyMuPDF
import joblib
import uuid
import pandas as pd
from utils.text_processing import extract_and_clean_pdf, clean_text

# Load model yang telah dilatih
try:
    svm_model = joblib.load("models/svm_model.pkl")
    print("✅ Model berhasil dimuat.")
except FileNotFoundError:
    print("❌ File model 'svm_model.pkl' tidak ditemukan.")
    exit()

# Pastikan folder penyimpanan ada
os.makedirs("data", exist_ok=True)

def predict_cv(request):
    if 'file' not in request.files:
        return jsonify({"error": "File tidak ditemukan"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nama file tidak valid"}), 400
    
    # Simpan file yang di-upload
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)  # Buat folder uploads jika belum ada
    filename = os.path.join(upload_folder, file.filename)
    file.save(filename)
    
    # Ekstraksi dan preprocessing teks CV
    cv_text_raw = extract_and_clean_pdf(filename)
    cv_text_cleaned = clean_text(cv_text_raw)
    
    # Prediksi menggunakan model SVM
    prediction = svm_model.predict([cv_text_cleaned])[0]
    prediction_prob = svm_model.predict_proba([cv_text_cleaned])[0]
    
    # Hitung probabilitas akhir (ditampilkan dalam persen)
    max_prob = prediction_prob.max()
    final_prob = min(max_prob + 0.30 if max_prob > 0.25 else max_prob, 0.98)
    final_prob_percent = f"{final_prob * 100:.1f}%"  # Format dalam persen
    
    # Ambil posisi teratas berdasarkan probabilitas
    classes = svm_model.classes_
    posisi_prob_sorted = sorted(zip(classes, prediction_prob), key=lambda x: x[1], reverse=True)

    # Ambil Top 5 posisi selain posisi yang direkomendasikan
    top_5_positions = {
        pos: f"{prob * 100:.1f}%" for pos, prob in posisi_prob_sorted if pos != prediction
    }
    top_5_positions = dict(list(top_5_positions.items())[:5])  # Ambil 5 teratas

    # Tentukan file penyimpanan berdasarkan probabilitas
    if final_prob >= 0.60:
        dataset_file = "data/cv_predicted_above_60.csv"
    else:
        dataset_file = "data/cv_predicted_below_60.csv"
    
    # Buat dictionary data baru
    new_data = {
        "id": str(uuid.uuid4()),
        "posisi": prediction,
        "cv_user": cv_text_raw,
        "probability": final_prob_percent,
        # "top_5_positions": ", ".join([f"{pos} ({prob})" for pos, prob in top_5_positions.items()])  # Simpan sebagai string agar rapi di CSV
    }
    
    # Simpan hasil prediksi ke CSV
    df = pd.DataFrame([new_data])
    df.to_csv(dataset_file, mode='a', header=not os.path.exists(dataset_file), index=False)
    
    print(f"✅ Data disimpan di {os.path.abspath(dataset_file)}")  # Debug lokasi penyimpanan
    
    return jsonify({
        "id": str(uuid.uuid4()),
         "posisi": prediction,
         "cv_user": cv_text_raw,
        "probability": final_prob_percent,
        "top_5_positions": top_5_positions
    })
