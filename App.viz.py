# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:40:32 2021

@author: bahas
"""

# -*- codi,ng: utf-8 -*-

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
import os
import xgboost
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from IPython.display import display
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.datasets import make_hastie_10_2
from sklearn.metrics import RocCurveDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pycaret.classification import *
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from pylab import *

# Set max columns to display
pd.options.display.max_columns = None


from plotly.subplots import make_subplots



# image1 = Image.open("logo.cities_growing.jpg")
# image2 = Image.open("logo.ubs.png")

# st.image([image2,    image1], use_column_width=False)


# Page 1 : affichage d'un texte simple
def accueil():
    
    st.title("Contexte")
    

        # Contexte des données
    contexte = """
    Les données COMPAS Scores sont un ensemble de données utilisé dans le domaine de la justice pénale pour prédire le risque de récidive des détenus aux états-unis. 
    Cet ensemble de données comprend des informations sur les caractéristiques des détenus, les scores de risque de récidive et d'autres variables liées au système de justice pénale. 
    Les scores COMPAS sont générés par un algorithme qui prend en compte divers facteurs, tels que l'âge, le sexe, les antécédents criminels, etc. 
    Ces scores sont utilisés par les professionnels du système de justice pour prendre des décisions concernant la probation, la libération conditionnelle, etc.

    L'exploration et l'analyse de ces données peuvent nous aider à mieux comprendre comment les scores COMPAS sont générés, à évaluer leur équité et à étudier les relations entre les caractéristiques des détenus et la récidive.

    """

    # Afficher le contexte dans Streamlit
    st.markdown(contexte,  unsafe_allow_html=True)

    # st.markdown("<description>  Cela fait référence à un ensemble de données qui est souvent utilisé dans le contexte de la justice pénale aux États-Unis. Le COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) est un algorithme développé pour évaluer le risque de récidive des délinquants" + 
    #             "Les données COMPAS comprennent généralement des informations sur les délinquants, telles que leur profil démographique, leurs antécédents criminels, leurs facteurs de risque, les décisions judiciaires précédentes, et les scores attribués par le système COMPAS. "+
    #             "Ces scores sont utilisés pour aider les professionnels de la justice à prendre des décisions sur la libération sous caution, la probation, la condamnation et d'autres aspects du processus judiciaire."+
    #             "Cependant, il est important de noter que l'utilisation de l'algorithme COMPAS a suscité des préoccupations en raison de ses éventuels biais et de ses impacts sur les disparités raciales. Il est essentiel de considérer ces problèmes lors de l'analyse et de l'interprétation des données COMPAS ou lors de l'utilisation de l'algorithme COMPAS dans le système judiciaire. " +
    #             "</description>", unsafe_allow_html=True)

        
DataSet_name=st.sidebar.selectbox("select DataSet", ("compas scores raw ", "compas scores raw online"))

# import pandas as pd
# compas_scores_raw = pd.read_csv("./archive/compas-scores-raw.csv")
# cox_violent_parsed = pd.read_csv("./archive/cox-violent-parsed.csv")
# cox_violent_parsed_filt = pd.read_csv("./archive/cox-violent-parsed_filt.csv")
# propublica_data_for_fairml = pd.read_csv("./archive/propublicaCompassRecividism_data_fairml.csv/propublica_data_for_fairml.csv")

# def get_dataset(DataSet_name):
#     if DataSet_name == "compas-scores-raw":
#         data = pd.read_csv("./archive/compas-scores-raw.csv")
#     else :
#         data = pd.read_csv("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
        
#     return data
# data = get_dataset(DataSet_name)

data = pd.read_csv("./archive/compas-scores-raw.csv")
#All texts that are African-American and African-Am should be the same.
data["Ethnic_Code_Text"] = data["Ethnic_Code_Text"].replace({"African-Am": "African-American"})
dat = data.drop(['AssessmentID', 'Case_ID', 'Agency_Text', 'LastName',
       'FirstName', 'MiddleName', 'ScaleSet_ID', 'Scale_ID', 'IsCompleted', 'IsDeleted'], axis=1)


# st.write("La taille de base de données", data.shape)



# Page 2 : affichage d'un graphique
def page_graphique():
    
    # Sélection des colonnes catégorielles
    x = data[['Agency_Text','Sex_Code_Text', 'Ethnic_Code_Text','ScaleSet',
        'Language', 'LegalStatus', 'CustodyStatus', 'MaritalStatus','RecSupervisionLevel', 
            'RecSupervisionLevelText','DisplayText', 'RawScore', 'DecileScore', 'ScoreText',
        'AssessmentType']]
   

    categorical_cols = x.select_dtypes(include=["object"]).columns.tolist()




    # Sélection de la variable et des couleurs
    var = st.selectbox("Sélectionner une variable :", categorical_cols)
    #color = st.color_picker("Choisir une couleur :", "#00f")
    color1, color2, color3, color4, color5, color6, color7, color8, color9, color10   = st.columns(10)#, 
    #  = st.columns(3)
    #  = st.columns(3)
    # = st.columns(1)
    color1 = color1.color_picker("Choisir la couleur 1 :", "#ff0000", key="color1")
    color2 = color2.color_picker("Choisir la couleur 2 :", "#00ff00", key="color2")
    color3 = color3.color_picker("Choisir la couleur 3 :", "#001BFF", key="color3")
    color4 = color4.color_picker("Choisir la couleur 4 :", "#FFFB00", key="color4")
    color5 = color5.color_picker("Choisir la couleur 5 :", "#201F1F", key="color5")
    color6 = color6.color_picker("Choisir la couleur 6 :", "#FF00F2", key="color6")
    color7 = color7.color_picker("Choisir la couleur 7 :", "#D09361", key="color7")
    color8 = color8.color_picker("Choisir la couleur 8 :", "#00FBFF", key="color8")
    color9 = color9.color_picker("Choisir la couleur 9 :", "#F5B9D5", key="color9")
    color10 = color10.color_picker("Choisir la couleur 10 :", "#00ff00", key="color10")


    colors = st.multiselect('Choisissez des couleurs',  [color1, color2, color3, color4, color5, color6, color7, color8, color9, color10])#


    # Création de l'histogramme
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = x[var].value_counts()
    bars = ax.bar(counts.index, counts, color= colors)
    ax.set_title(var)
    ax.tick_params(axis='x', rotation=45)

    # Ajout d'une légende pour les couleurs
    if len(counts) > 1:
        ax.legend(bars, counts.index, bbox_to_anchor=(0.2, 1), loc='upper left')

    # Affichage de l'histogramme
    st.pyplot(fig)



    # Sélection des colonnes numériques
    numeric_cols = x.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Sélection de la variable et des couleurs
    var = st.selectbox("Sélectionner une variable :", numeric_cols)
    color = st.color_picker("Choisir une couleur :", "#00f")

    # Création de l'histogramme
    fig, ax = plt.subplots()
    sns.histplot(data[var], color=color, ax=ax)
    ax.set_title(var)

    # Affichage de l'histogramme
    st.pyplot(fig)


        
    # Sélection des colonnes catégorielles
    cat_cols = data.select_dtypes(include=["object"]).columns.tolist()

    # Sélection des colonnes quantitatives
    quant_cols = data.select_dtypes(include=["float", "int"]).columns.tolist()

    # Affichage des menus déroulants pour les variables catégorielles et quantitatives à afficher
    selected_cat_col = st.selectbox("Choisissez une variable catégorielle :", cat_cols)
    selected_quant_col = st.selectbox("Choisissez une variable quantitative :", quant_cols)

    # Affichage des diagrammes en boîte ou en barres pour les variables sélectionnées
    plot_type = st.selectbox("Choisissez le type de graphique :", ["Boxplot", "Barplot"])


    if plot_type == "Boxplot":
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x=selected_cat_col, y=selected_quant_col, data=data)
        # Configuration des étiquettes de l'axe x
        plt.xticks(rotation=45, ha="right")
        plt.title(selected_cat_col + ' vs. ' + selected_quant_col)
        st.pyplot(fig)
    elif plot_type == "Barplot":
        fig, ax = plt.subplots(figsize=(12, 8))
        agg_data = data.groupby(selected_cat_col)[selected_quant_col].mean()
        agg_data.plot(kind="bar")
        # Configuration des étiquettes de l'axe x
        plt.xticks(rotation=45, ha="right")
        plt.title(selected_cat_col + ' vs. ' + selected_quant_col)
        st.pyplot(fig)






    # Sélection des variables
    var1 = st.selectbox("Choisissez la première variable :", data.columns)
    var2 = st.selectbox("Choisissez la deuxième variable :", data.columns)

    # Sélection de la couleur
    # color = st.selectbox("Choisissez une couleur :", sns.color_palette())
    # Choisir la palette de couleurs à utiliser
    #palette = st.selectbox("Choisissez une palette de couleurs :", sns.color_palette().as_hex(), key="palette")
    color1, color2, color3, color4, color5, color6, color7, color8, color9, color10   = st.columns(10)#, 

    color1 = color1.color_picker("Choisir la couleur 1 :", "#ff0000", key="col1")
    color2 = color2.color_picker("Choisir la couleur 2 :", "#00ff00", key="col2")
    color3 = color3.color_picker("Choisir la couleur 3 :", "#001BFF", key="col3")
    color4 = color4.color_picker("Choisir la couleur 4 :", "#FFFB00", key="col4")
    color5 = color5.color_picker("Choisir la couleur 5 :", "#201F1F", key="col5")
    color6 = color6.color_picker("Choisir la couleur 6 :", "#FF00F2", key="col6")
    color7 = color7.color_picker("Choisir la couleur 7 :", "#D09361", key="col7")
    color8 = color8.color_picker("Choisir la couleur 8 :", "#00FBFF", key="col8")
    color9 = color9.color_picker("Choisir la couleur 9 :", "#F5B9D5", key="col9")
    color10 = color10.color_picker("Choisir la couleur 10 :", "#00ff00", key="col10")

    colors = st.multiselect('Choisissez des couleurs',  [color1, color2, color3, color4, color5, color6, color7, color8, color9, color10], key="code_col1")#

    # Croisement des variables
    grouped_data = data.groupby([var1, var2]).size().reset_index(name='count')

    # Création du graphique en barres
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=var1, y="count", hue=var2, data=grouped_data, palette=colors)
    plt.title("Croisement de {} et {} par {}".format(var1, var2, colors))
    plt.xlabel(var1)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)



    # Affichage des menus déroulants pour les variables catégorielles à afficher
    selected_cat_col1 = st.selectbox("Choisissez une variable catégorielle :", cat_cols, key="cat_col1")
    selected_cat_col2 = st.selectbox("Choisissez une autre variable catégorielle :", cat_cols, key="cat_col2")

    # Création du tableau de contingence
    contingency_table = pd.crosstab(data[selected_cat_col1], data[selected_cat_col2])

    # Affichage du tableau de contingence
    st.write(contingency_table)

    # Affichage du heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(contingency_table, annot=True, cmap='Blues')
    plt.title(selected_cat_col1 + ' vs. ' + selected_cat_col2)
    plt.show()
    st.pyplot(fig)


# Page 4 : affichage d'un formulaire
def ML():
    # image1 = Image.open("forces_01.jpg")
    
    st.image("shap_001.png", caption='Légende de l\'image')
    st.image("forces_01.jpg", caption='Légende de l\'image')





# Création du menu latéral
choix_page = st.sidebar.radio("Sélectionnez une page :", ("Accueil", "Graphique", 'ML'))

# Affichage de la page sélectionnée
if choix_page == "Accueil":
    accueil()
elif choix_page == "Graphique":
    page_graphique()
else:
    ML()





