from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def train_and_evaluate_model(X, y):
    """
    Model eğitme ve test seti üzerinde değerlendirme.

    Args:
    X (sparse matrix): Özellik matrisi
    y (array): Sınıf etiketleri
    """
    try:
        # SMOTE ile veri dengeleme (bazı sınıflarda az veri olduğu için evaluate aşamasında 0 elde ediyorduk)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print("Data balanced successfully")

        # Eğitim ve test verilerine ayırma
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
        print("dataset splitted into train and test sets")

        # Logistic Regression modelini oluşturma
        model = LogisticRegression(class_weight='balanced', random_state=42)

        # Hyperparameter tuning için GridSearchCV kullanma
        param_grid = {
            'C': [0.1, 1, 10],  # Regularization strength
            'max_iter': [100, 200, 300]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # En iyi parametrelerle modeli eğitme
        best_model = grid_search.best_estimator_

        # Test seti üzerinde tahmin yapma
        y_pred = best_model.predict(X_test)

        # Sonuçları değerlendirme
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Confusion Matrix grafiği
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y), yticklabels=set(y))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

        # Precision, Recall, F1-Score grafiklerini çizme
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = ['precision', 'recall', 'f1-score']
        labels = list(report.keys())[:-3]  # Sonuçta last 3 label (support vs.) olduğu için çıkarıyoruz

        # Bar grafik oluşturma
        for metric in metrics:
            values = [report[label][metric] for label in labels]
            plt.figure(figsize=(8, 6))
            plt.bar(labels, values, color='lightblue')
            plt.title(f'{metric.capitalize()} for each class')
            plt.xlabel('Class')
            plt.ylabel(f'{metric.capitalize()}')
            plt.show()

    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        raise
