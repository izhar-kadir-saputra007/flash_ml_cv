from routes.prediction_routes import prediction_bp
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import PyPDF2
import os

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS

# Mengunduh data NLTK jika belum ada
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Memuat model dan vectorizer yang telah disimpan
model = joblib.load('models/svm_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

# Inisialisasi lemmatizer dan stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Membersihkan dan memproses teks resume.
    """
    try:
        # Menghapus karakter unik (non-ASCII)
        text = text.encode('ascii', errors='ignore').decode()
        # Mengubah teks menjadi huruf kecil
        text = text.lower()
        # Menghapus tanda baca
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenisasi
        tokens = text.split()
        # Menghapus stopwords dan melakukan lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        # Menggabungkan kembali token menjadi string
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        return ""

def extract_text_from_pdf(file_path):
    """
    Mengekstrak teks dari file PDF.
    """
    text = ""
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + " "
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

@app.route('/')
def home():
    return "✅ API CV Prediction sudah aktif!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk melakukan prediksi kategori pekerjaan.
    Menerima data dalam bentuk teks atau file PDF.
    """
    try:
        # Cek apakah ada file yang diunggah
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            # Simpan file sementara
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            # Ekstrak teks dari PDF
            resume_text = extract_text_from_pdf(file_path)
            # Hapus file setelah ekstraksi
            os.remove(file_path)
        elif 'resume_text' in request.json:
            resume_text = request.json['resume_text']
        else:
            return jsonify({'error': 'No resume data provided.'}), 400

        # Pra-pemrosesan teks resume
        clean_text = preprocess_text(resume_text)

        # Vectorisasi teks resume
        resume_vector = vectorizer.transform([clean_text])

        # Melakukan prediksi
        predicted_category = model.predict(resume_vector)[0]
        probabilities = model.predict_proba(resume_vector)[0]
        prob_index = list(model.classes_).index(predicted_category)
        probability = probabilities[prob_index] * 100  # Mengubah ke persen

        # Menyiapkan probabilitas untuk semua kelas
        probabilities_dict = {cls: round(prob * 100, 2) for cls, prob in zip(model.classes_, probabilities)}

        # Menyiapkan respon
        response = {
            'predicted_category': predicted_category,
            'probability': round(probability, 2),
            'probabilities': probabilities_dict
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500
    
    # Konfigurasi upload file
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Register blueprint
app.register_blueprint(prediction_bp, url_prefix='/api')


if __name__ == '__main__':
    # Membuat folder uploads jika belum ada
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Menjalankan aplikasi Flask
    app.run(host='0.0.0.0', port=5000)
