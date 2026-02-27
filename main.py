import pandas as pd
from preprocess import preprocess
from featureExtraction import extract_features
from train import train_and_evaluate_model


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def main():

    filepath = 'C:\\Users\\gulco\\Desktop\\finalProject\\PROMISE_exp.csv'
    df = load_data(filepath)
    print("Data uploaded. first 5 row:")
    print(df.head())

    # Preprocessing
    print("\n preprocessing steps...")
    df['RequirementText'] = df['RequirementText'].dropna().apply(preprocess)
    print("preprocessing is done. first 5 processed row:")
    print(df.head())

    # Özellik çıkarma
    print("\n feature etraction ..")
    X, vectorizer = extract_features(df)
    y = df['_class_']
    print("feature extraction is done. feature matrix shape:", X.shape)

    # Model eğitme ve değerlendirme
    print("\n training and evaluating ..")
    train_and_evaluate_model(X, y)


if __name__ == "__main__":
    main()
