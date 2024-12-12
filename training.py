from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from featureExtraction import extract_features
from preprocessing import df
from imblearn.over_sampling import SMOTE

# Training: Naïve Bayes
def train_naive_bayes(X, y):
    # SMOTE ile verinizi dengeleyin
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Eğitim ve test verilerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Initialize the Naive Bayes classifier
    model = MultinomialNB()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Preprocess and extract features
X, vectorizer = extract_features(df)
y = df['_class_']  # Target labels

# Train and evaluate the model
train_naive_bayes(X, y)
