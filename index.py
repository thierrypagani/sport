
import pandas as pd
import numpy as np
import matplotlib as plt
import streamlit as st
import altair as alt
from PIL import Image
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import r2_score

Athletes=pd.read_csv("../datas/Athletes.csv")

st.sidebar.title("Sommaire")

pages = ["Données", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0]:
    st.write("### Données")
    
    st.dataframe(Athletes.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(Athletes.shape)
    
    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(Athletes.isna().sum())
        
    if st.checkbox("Afficher les doublons") : 
        st.write(Athletes.duplicated().sum())

elif page == pages[1]:
    st.write("### Exploration des données")

    if st.checkbox("### Afficher les colonnes"):
       st.write(Athletes.columns)  
    
    if st.checkbox("### Afficher les catégories"):
       st.write(Athletes['Sport'].unique().tolist())
      
    if st.checkbox("### Afficher les Pays"):
       st.write(Athletes['Nation'].unique().tolist())

elif page == pages[2]:
    st.write("### Analyse des données")

    if st.checkbox("### Joueur le plus grand"):
       st.write(Athletes[Athletes["Height (cm)"]== Athletes["Height (cm)"].max()] )
    
    if st.checkbox("### Joueur le plus petit"):
       st.write(Athletes[Athletes["Height (cm)"]== Athletes["Height (cm)"].min()] )
      
    if st.checkbox("### Joueur le moins bien payé"):
       st.write(Athletes[Athletes["Total Pay"]== Athletes["Total Pay"].max()]) 
    
    if st.checkbox("### Joueur le mieux payé"):
       st.write(Athletes[Athletes["Total Pay"]== Athletes["Total Pay"].min()])

    if st.checkbox("### Joueur le plus vieu"):
       st.write(Athletes[Athletes["Year of birth"]== Athletes["Year of birth"].min()])

    if st.checkbox("### Joueur le plus jeune"):
       st.write(Athletes[Athletes["Year of birth"]== Athletes["Year of birth"].max()])
