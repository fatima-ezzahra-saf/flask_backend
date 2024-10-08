import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_models():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Vérifier l'extension du fichier
    ext = os.path.splitext(file.filename)[1].lower()
    try:
        if ext in ['.xls', '.xlsx']:  # Vérifie si le fichier est un fichier Excel
            df = pd.read_excel(file, engine='openpyxl')  # Lire le fichier Excel avec openpyxl
        elif ext in ['.csv']:  # Vérifie si le fichier est un fichier CSV
            df = pd.read_csv(file, sep=';')  # Lire le fichier CSV en spécifiant le séparateur
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Récupérer les noms des variables explicatives et la variable cible à partir de la requête
    explicative_vars = request.form.get('explicative_vars').split(',')
    target_var = request.form.get('target_var')

    # Vérifier que les variables sont présentes dans le DataFrame
    if not all(var in df.columns for var in explicative_vars) or target_var not in df.columns:
        return jsonify({'error': 'Invalid variable names provided'}), 400

    # Séparer les variables explicatives et la variable cible
    X = df[explicative_vars]
    y = df[target_var]

    # Prétraitement des données
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()  # Variables numériques
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()  # Variables catégorielles

    # Créer des pipelines pour le prétraitement
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))  # Impute la moyenne pour les données numériques
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Impute une valeur constante pour les données catégorielles
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Convertit les catégories en indicateurs
    ])

    # Combinez les transformations dans un préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer un pipeline complet pour les modèles
    models = {
        'Logistic Regression': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))]),
        'SVM': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC())]),
        'Random Forest': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
    }

    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)

    return jsonify(accuracies)

if __name__== '__main__':
    app.run(debug=True)