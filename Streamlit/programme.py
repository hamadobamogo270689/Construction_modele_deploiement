import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

# Charger le dataset
data = pd.read_csv('tweets_suspect.csv')  # 'fichier des tweets suspects

# Gestion du déséquilibre des classes par sous-échantillonnage
data_majoritaire = data[data['label'] == 1]  # Classe non suspecte (majoritaire)
data_minoritaire = data[data['label'] == 0]  # Classe suspecte (minoritaire)

# Sous-échantillonnage de la classe majoritaire
data_majoritaire_sous = resample(data_majoritaire,
                                 replace=False,    # échantillonnage sans remplacement
                                 n_samples=len(data_minoritaire),  # égaliser au nombre de la classe minoritaire
                                 random_state=42)

# Combiner les deux classes après équilibrage
data_equilibree = pd.concat([data_majoritaire_sous, data_minoritaire])

# Vérification de la nouvelle répartition des classes
print(data_equilibree['label'].value_counts())

"""##  Nettoyage du texte avec SpaCy"""

# Charger le modèle SpaCy anglais
nlp = spacy.load("en_core_web_sm")

# Nettoyage du texte avec SpaCy
def nettoyer_texte_spacy(texte):
    doc = nlp(texte.lower())  # Convertir en minuscules et traiter avec SpaCy
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Appliquer le nettoyage sur chaque texte
data_equilibree['message_nettoye'] = data_equilibree['message'].apply(nettoyer_texte_spacy)

"""## Vectorisation"""

# Vectorisation avec TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data_equilibree['message_nettoye']).toarray()
# Ajuster le TF-IDF sur l'ensemble d'entraînement
tfidf = TfidfVectorizer(max_features=5000)  # Utiliser 5000 caractéristiques comme dans l'entraînement
#tfidf.fit(df_train['tweet_text'])  # Remplacez 'tweet_text' par la colonne contenant les tweets

# Sauvegarder le TF-IDF ajusté
joblib.dump(X, "tfidf_vectorizer.pkl")
y=data_equilibree['label'].values
# Encodage de la variable cible
#le = LabelEncoder()
#y = le.fit_transform(data['message'])



"""## Entrainement du modèle"""

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



# Entraînement et évaluation des modèles
# Définition des modèles avec gestion des meilleurs paramètres spécifiés
models = {
    'Logistic Regression': LogisticRegression(C=10),
    'Random Forest': RandomForestClassifier(max_depth=None, n_estimators=100),
    'SVM': SVC(C=1, kernel='linear')  # Meilleurs paramètres spécifiés pour SVM
}

# Entraînement, évaluation et sauvegarde des modèles
for name, model in models.items():
    model.fit(X_train, y_train)
    # Sauvegarde du modèle
    filename = f"{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, filename)
#-------------------------------------------------------------------------#