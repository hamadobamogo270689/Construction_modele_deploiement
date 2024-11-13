# Construction_modele_deploiement
Le projet vise à classifier des tweets en "suspects" et "non suspects" en utilisant des données équilibrées via le sous-échantillonnage pour éviter les biais. Le prétraitement des tweets est réalisé avec SpaCy. Les modèles évalués incluent la régression logistique, la forêt aléatoire et le SVM.
### Prérequis
Pour exécuter ce projet, assurez-vous d'avoir installé les dépendances suivantes :
Python 3.x, Pandas, Matplotlib, Seaborn, spaCy, scikit-learn.
### 1. Structure du Code
#### 1.1. Importation des Bibliothèques
Dans cette section, nous importons toutes les bibliothèques nécessaires pour la manipulation des données, la visualisation, et les algorithmes d'apprentissage automatique.
#### 1.2. Chargement et Exploration des Données
Cette section charge le fichier des tweets suspects(voir dossier dataset), vérifie la structure des données, et effectue quelques visualisations exploratoires pour comprendre les classes des tweets.
##### Equilibrage des données
Le sous echantillonnage est utilisé qpour ré-equilibré des données
#### 1.3. Prétraitement des Données

- La suppression des caractères spéciaux avec spaCy
- La tokenisation avec spaCy
- La vectorisation des données textuelles avec TF-IDF

#### 1.4. Division des Données
Utilisation de train_test_split pour séparer les données en ensembles d'entraînement et de test.
#### 1.5. Entraînement du Modèle
Application des algorithmes de classification et comparaison de leurs performances :

- *SVM*
- *Random Forest*
- *K-Nearest Neighbors*, etc.
#### 1.6. Ajustement des hyperparamètres 
Application de ces algorithmes de classification avec les hyperparamètres et comparaison de leurs performances
#### 1.7. Évaluation du Modèle
Présentation des métriques de performance, telles que l'accuracy, la précision, les matrices de confusion et le rappel, pour évaluer la qualité des modèles.
### 2. Streamlit
- *programme.py* : permet d"enregistrer et sauvegarder nos trois modèles(Logistic regression, random forest et SVM) afin de les chargés dans run_streamlit.py
- *run_streamlit.py* : Script principal pour exécuter l'application Streamlit. Ce fichier lance l'interface utilisateur, permettant de charger et tester les modèles de classification (régression logistique, forêt aléatoire, SVM) sur des tweets pour déterminer s'ils sont suspects ou non.

- *logistic_regression_model.pkl* : Modèle de régression logistique pré-entraîné notre fichier * programme.py*, enregistré au format .pkl. 
- *random_forest_model.pkl* : Modèle de forêt aléatoire pré-entraîné notre fichier * programme.py*, également enregistré au format .pkl. 

- *svm_model.pkl* : Modèle de Support Vector Machine (SVM) pré-entraîné dans notre fichier * programme.py* , enregistré dans un fichier .pkl.
- *tfidf_vectorizer.pkl: TF-IDF pré-entraîné dans notre fichier * programme.py**, enregistré dans un fichier .pkl.
### 3. Figures
#### 3.1 Analyse univariée
- *hist.png* représente l'histogramme de notre dataset
- *circ.png* est le diagramme circulaire de notre dataset
#### 3.2 Résultats des métriques des trois modèles
##### 3.2.1 Sans inclusion des hyperparamètres
- *logistic.png* représente le résulat des différentes métriques de la logistic regression
- *random.png* est le résulat des différentes métriques de random forest
- *svm.png* est le résulat des différentes métriques de svm

##### 3.2.2 Avec inclusion des hyperparamètres
- *logistic_Dataequilibre.png* représente le résulat des différentes métriques avec les hyperparamètres de la logistic regression
- *rf_Dataequilibre.png* est le résulat des différentes métriques avec les hyperparamètres de random forest
- *svm_Dataequilibre.png* est le résulat des différentes métriques avec les hyperparamètres de svm
- #### 3.3 Matrix de confusion des trois modèles
##### 3.3.1 Sans inclusion des hyperparamètres
- *matlo.png* représente la matrice de confusion de la logistic regression
- *matrf.png* est  la matrice de confusion de random forest
- *matsv.png* est  la matrice de confusion de svm
##### 3.3.2 Avec inclusion des hyperparamètres
- *matlog_Dataequilibre.png* représente la matrice de confusion de la logistic regression avec les hyperparamètres
- *matrf_Dataequilibre.png* est  la matrice de confusion de random forest avec les hyperparamètres
- *matsvm_Dataequilibre.png* est  la matrice de confusion de svm avec les hyperparamètres
### 4. Dataset
  - *tweet_suspects.csv* est le dataset utilisé pour la classification
