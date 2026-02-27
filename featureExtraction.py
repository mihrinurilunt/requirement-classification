from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def extract_features(df):
    """
    Extracts features using Bag-of-Words.
    Returns:
    X (sparse matrix): Feature matrix.
    """
    vectorizer = CountVectorizer()  #kelime sıklıklarını temel alarak bir 'Bag-of-Words' features çıkarır

    # Fit and transform the RequirementText column
    X = vectorizer.fit_transform(df['RequirementText'])
    feature_names = vectorizer.get_feature_names_out()

    print(f"Number of features: {len(feature_names)}")  #benzersiz kelime sayısını bulacak
    print(f"Sample features: {feature_names[:10]}")
    return X,vectorizer


"""
CountVectorizer:
Kelime sıklıklarını temel alarak bir "Bag-of-Words" (BoW) özelliği çıkarır.

Çıktının boyutu (ör. (626, 1200)), requirements metinlerinin sayısı (satır) ve kullanılan kelime sayısını (sütun) temsil eder.
"""
