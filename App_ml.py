import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from IPython.display import display

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import os


from sklearn.preprocessing import LabelEncoder
import xgboost
import shap

# Définition de la fonction qui calcule l'âge en années
def dateOfBirthTransform(P,G):
    formatting = "%m/%d/%Y"
    x = P.split("/")
    if x[-2] == "0":
        x[-1] = "20" + x[-1][-2:]
    else:
        x[-1] = "19" + x[-1][-2:]
    date_string = "/".join(x)
    date2 = datetime.strptime(G, '%m/%d/%y %H:%M')
    date_object = datetime.strptime(date_string, formatting)
    age = (date2 - date_object).days // 365
    return int(age)

# Charger les données
compas_scores_raw = pd.read_csv("./archive/compas-scores-raw.csv")

# Appliquer la transformation de l'âge
compas_scores_raw['years'] = compas_scores_raw.apply(lambda row: dateOfBirthTransform(row["DateOfBirth"], row["Screening_Date"]), axis=1)

#Suppression des données manquantes 
compas_scores_raw = compas_scores_raw.drop(['AssessmentID', 'Case_ID',  'LastName',
       'FirstName', 'MiddleName', 'DateOfBirth', 'ScaleSet_ID', 'ScaleSet', 'AssessmentReason',
       'Screening_Date', 'Scale_ID',  'AssessmentType', 'IsCompleted', 'IsDeleted', ], axis=1)

# Appliquer le one-hot encoding à la variable "Ethnic_Code_Text"
one_hot_encoded_ethnic = pd.get_dummies(compas_scores_raw['Ethnic_Code_Text'], prefix='Ethnic')

# Fusionner les données encodées avec le dataframe d'origine
compas_scores_encoded = pd.concat([compas_scores_raw, one_hot_encoded_ethnic], axis=1)

# Variables nécessitant le label encoding
label_encoding_cols = ['Sex_Code_Text', 'DisplayText', 'Language', 'ScoreText', 'RecSupervisionLevelText']

# Variables nécessitant le one-hot encoding
one_hot_encoding_cols = ['LegalStatus', 'CustodyStatus', 'MaritalStatus', 'Agency_Text']

# Appliquer le label encoding
label_encoder = LabelEncoder()
for col in label_encoding_cols:
    compas_scores_encoded[col] = label_encoder.fit_transform(compas_scores_raw[col])

# Appliquer le one-hot encoding
one_hot_encoded_data = pd.get_dummies(compas_scores_encoded[one_hot_encoding_cols], drop_first=True)

# Fusionner les données encodées
compas_scores_encoded = pd.concat([compas_scores_encoded.drop(one_hot_encoding_cols, axis=1), one_hot_encoded_data], axis=1)

compas_scores_encoded = compas_scores_encoded.drop(columns=['Ethnic_Code_Text', 'Ethnic_Oriental'])

# Afficher les données encodées
st.write(compas_scores_encoded.head())

# Supprimer les lignes avec des valeurs manquantes dans toutes les colonnes
data_without_missing = compas_scores_encoded.dropna()

# # Afficher la matrice de covariance
# st.pyplot(sns.heatmap(compas_scores_encoded.corr()))
# Afficher la matrice de covariance
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(compas_scores_encoded.corr(), ax=ax)
plt.title('Matrice de covariance')
plt.tight_layout()

# Enregistrer le graphique en tant qu'image temporaire
temp_file = 'temp_covariance.png'
plt.savefig(temp_file)

# Afficher l'image dans Streamlit
st.image(temp_file)

# Supprimer le fichier temporaire
os.remove(temp_file)



# Sélection des variables X et y
# X = ...  # Vos données d'entraînement
# y = ...  # Vos étiquettes cibles

X = compas_scores_encoded.drop(columns=["DisplayText",'Person_ID', 'RecSupervisionLevelText','DecileScore', 'RawScore', 'ScoreText', 'RecSupervisionLevelText'])
y = compas_scores_encoded['DecileScore']


# Split des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement du modèle GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# Prédiction sur l'ensemble de test
y_pred = clf.predict(X_test)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Affichage de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs réelles')
plt.title('Matrice de confusion')
st.pyplot()


# Split des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement du modèle GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(X_train, y_train)

# Obtention des scores de probabilité prédits pour chaque classe dans l'ensemble de test
y_scores = clf.predict_proba(X_test)

# Calcul des courbes ROC pour chaque classe
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(clf.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_scores[:, i], pos_label=clf.classes_[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Affichage des courbes ROC
plt.figure(figsize=(8, 6))

for i in range(len(clf.classes_)):
    plt.plot(fpr[i], tpr[i], label='Courbe ROC (Classe {}) (AUC = {:.2f})'.format(clf.classes_[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC - Classification Multiclasse')
plt.legend(loc='lower right')

# Affichage de la figure dans Streamlit
st.pyplot(plt)



# Split des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convertir les étiquettes en entiers dans la plage attendue
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Créer et entraîner le modèle XGBoost
model = xgboost.XGBClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model.fit(X_train, y_train_encoded)

# Faire des prédictions sur l'ensemble de test
y_pred_encoded = model.predict(X_test)

# Convertir les prédictions en étiquettes d'origine
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Affichage de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs réelles')
plt.title('Matrice de confusion')

# Affichage de la figure dans Streamlit
st.pyplot(plt)



# Obtenir les scores de probabilité prédits pour chaque classe dans l'ensemble de test
y_scores = model.predict_proba(X_test)

# Calculer les courbes ROC pour chaque classe
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(model.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_scores[:, i], pos_label=model.classes_[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Tracer les courbes ROC pour chaque classe
plt.figure(figsize=(8, 6))

for i in range(len(model.classes_)):
    plt.plot(fpr[i], tpr[i], label='Courbe ROC (Classe {}) (AUC = {:.2f})'.format(model.classes_[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC - Classification Multiclasse')
plt.legend(loc='lower right')

# Affichage de la figure dans Streamlit
st.pyplot(plt)



# Créer l'explainer SHAP
explainer_compas_score = shap.Explainer(model)

# Calculer les valeurs SHAP
shap_values_compas = explainer_compas_score.shap_values(X_test)

# Afficher les contributions des variables
fig = shap.summary_plot(shap_values_compas, X_test, plot_type="bar")

# Affichage de la figure dans Streamlit
st.pyplot(fig)

