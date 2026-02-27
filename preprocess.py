import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd


def preprocess(text):
    try:
        if pd.isna(text) or not isinstance(text, str):  #eğer text boşsa
            return ""

        text = text.lower()

        # Gereksiz karakterleri ve sayıları kaldırma
        text = re.sub(r'[^a-z\s]', '', text)  # Sadece harfleri ve boşlukları bırakır

        tokens = word_tokenize(text)  # Tokenization

        stop_words = set(stopwords.words('english'))
        custom_stop_words = {'shall', 'must', 'may', 'could'}
        stop_words = stop_words.union(custom_stop_words)

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        cleaned_tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word.isalpha() and word not in stop_words
        ]

        return ' '.join(cleaned_tokens) # Tokenleri tekrar birleştirme

    except Exception as e:
        print(f"Error processing text: {text}, Error: {str(e)}")
        return ""

"""
1. Text Normalization & Cleaning
Büyük/küçük harf dönüşümü (hepsini küçük harfe çevirme).
Noktalama işaretlerini, özel karakterleri veya sayıları kaldırma.
Gereksiz boşlukları temizleme.

2. Tokenization
Metni kelime veya cümle bazında bölme.

3. Lemmatization
Kelimeleri köklerine (sözlük formuna) indirme.

4. Stemming
Kelimelerin son eklerini keserek köke indirme (daha agresif bir yöntem).
"""
