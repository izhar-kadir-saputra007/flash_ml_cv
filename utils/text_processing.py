import fitz  # PyMuPDF
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory



# Coba download 'punkt_tab' dan tangani jika tidak tersedia
try:
    nltk.download("punkt_tab")
except:
    print("punkt_tab tidak ditemukan di NLTK, tetapi program tetap berjalan.")

# Pastikan resource NLTK sudah diunduh sebelumnya
try:
    stop_words = frozenset(stopwords.words("indonesian"))
except:
    nltk.download("stopwords")
    nltk.download("punkt")
    stop_words = frozenset(stopwords.words("indonesian"))

# Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    """ Membersihkan teks dari simbol dan stopwords, lalu melakukan stemming """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Hilangkan karakter non-alfanumerik
    text = re.sub(r"\s+", " ", text).strip()  # Hilangkan spasi berlebih
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def extract_and_clean_pdf(pdf_path):
    """ Mengekstrak teks dari PDF dan hanya membersihkan informasi pribadi """
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text("text") for page in doc])
    return remove_personal_info(text)

def remove_personal_info(text):
    """ Menghapus informasi pribadi seperti email, nomor HP, dan alamat """
    text = re.sub(r'\S+@\S+', '', text)  # Hapus email
    text = re.sub(r'(\+62|62|0)\d{9,12}', '', text)  # Hapus nomor HP
    text = re.sub(r'\b(jalan|jl\.?|alamat)\b', '', text, flags=re.IGNORECASE)  # Hapus alamat
    text = re.sub(r'\bnama\b', '', text, flags=re.IGNORECASE)  # Hapus kata 'nama'
    text = re.sub(r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b', '', text)  # Hapus tanggal
    text = re.sub(r'@\w+', '', text)  # Hapus @username
    text = re.sub(r'\b(?:twitter|facebook|linkedin|instagram)\.com\/\S+\b', '', text, flags=re.IGNORECASE)  # Hapus sosial media
    text = re.sub(r'[\(\)\[\]\{\}]', '', text)  # Hapus tanda kurung
    text = re.sub(r'[-=•●▪▸▶►]', '', text)  # Hapus simbol tertentu
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    return text
