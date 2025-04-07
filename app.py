from flask import Flask, request, jsonify
from flask_cors import CORS
from routes.prediction_routes import prediction_bp
import joblib
import nltk
import string
import os
import PyPDF2
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Konfigurasi folder upload
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Register blueprint
app.register_blueprint(prediction_bp, url_prefix='/api')

# Setup model & vectorizer
model = joblib.load('models/svm_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

# Setup NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Utils
def preprocess_text(text):
    try:
        text = text.encode('ascii', errors='ignore').decode().lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        return ""

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

@app.route('/')
def home():
    return "✅ API CV Prediction sudah aktif!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            resume_text = extract_text_from_pdf(file_path)
            os.remove(file_path)
        elif 'resume_text' in request.json:
            resume_text = request.json['resume_text']
        else:
            return jsonify({'error': 'No resume data provided.'}), 400

        clean_text = preprocess_text(resume_text)
        resume_vector = vectorizer.transform([clean_text])
        predicted_category = model.predict(resume_vector)[0]
        probabilities = model.predict_proba(resume_vector)[0]
        prob_index = list(model.classes_).index(predicted_category)
        probability = probabilities[prob_index] * 100
        probabilities_dict = {cls: round(prob * 100, 2) for cls, prob in zip(model.classes_, probabilities)}

        response = {
            'predicted_category': predicted_category,
            'probability': round(probability, 2),
            'probabilities': probabilities_dict
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

