import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import plotly.graph_objects as go

import shap
#shap.initjs()


import requests
from requests.exceptions import HTTPError
import json

url = "https://fastapiidreindex.herokuapp.com/predict"

st.set_page_config(
    page_title="Prêt à dépenser - Un algorithme de classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("""
Cette app calcule la probabilité qu’un client rembourse son crédit.
""")

#Former la barre latérale
st.sidebar.header("Données personnelles du client (ID)")
st.header("Analyse de décision")


def user_input_features():
    
    SK_ID_CURR = st.sidebar.number_input('ID du prêt', 100001, 456250)

    return {'SK_ID_CURR': SK_ID_CURR}

data_in = user_input_features()

df = pd.DataFrame([data_in])
st.sidebar.write(df)

#st.sidebar.write(data_in)

if st.sidebar.button('Predict'):

    try:
        response = requests.post(url = url, data = json.dumps(data_in))
        st.sidebar.text(f"API status code: {response.status_code}")
        response.raise_for_status()
    except HTTPError as http_err:
        st.sidebar.text(f"HTTP error occurred: {http_err}")
        st.sidebar.text(f"{response.text}")
    except Exception as err:
        st.sidebar.text(f"error occurred: {err}")
    else:
        st.sidebar.text(f"API answered: {response.json()}")


# S3
import boto3
from io import BytesIO
s3 = boto3.client('s3')
bucket_name = 'fastapimodels2' # to be replaced with bucket name


# Initialize files
import joblib
def read_s3_joblib_file(key):
    with BytesIO() as data:
        s3.download_fileobj(Fileobj=data, Bucket=bucket_name, Key=key)
        data.seek(0)
        return joblib.load(data)

def read_s3_csv_file(key):
    with BytesIO() as data:
        s3.download_fileobj(Fileobj=data, Bucket=bucket_name, Key=key)
        data.seek(0)
        return pd.read_csv(data)

df_test = read_s3_csv_file('test_ID_reindex.csv')
random_forest = read_s3_joblib_file('random_forest_best.joblib')


#Informations clients
st.subheader("Informations clients")

df_client = df.merge(right=df_test, on = 'SK_ID_CURR', how = 'inner')
df_client = df_client.reindex(df_test.columns, axis=1)
st.write(df_client)

if df_client.SK_ID_CURR.empty:
    st.error("Il n'y a pas de client avec cet ID.")
else:


    # Score du client
    st.subheader("Score du client")
    st.caption("Évaluation du client, qui permet de juger s'il est loin du seuil de remboursement du prêt ou non.")

    threshold = 0.680000
    df_client_2 = df_client.set_index('SK_ID_CURR')

    probability = random_forest.predict_proba(df_client_2)
    st.write(probability)

    probability_2 = random_forest.predict_proba(df_client_2).max()

    
    fig_1 = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability_2,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Score", 'font': {'size': 24}},
        delta = {'reference': threshold, 'decreasing': {'color': "green"},
                 'increasing': {'color' : 'red'}},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "silver"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.5], 'color': 'yellowgreen'},
                {'range': [0.5, threshold], 'color': 'orange'},
                {'range':[threshold,1], 'color':'red'}],
            'threshold': {
                'line': {'color': "firebrick", 'width': 4},
                'thickness': 0.75,
                'value': threshold}}))

    fig_1.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
    st.plotly_chart(fig_1)


    #Variables affectant la décision
    st.subheader("Variables affectant la décision")
    st.subheader ('Graphique №1')
    st.caption("Histogramme d'importance variable classique")

    df_test_2=df_test.set_index('SK_ID_CURR')

    explainer = shap.KernelExplainer(random_forest.predict, shap.sample(df_test_2,50))
    shap_values = explainer.shap_values(df_client_2)

    #matrix=pd.DataFrame(shap_values)
    #st.write(matrix)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig_2 = shap.summary_plot(shap_values, df_client_2, plot_type="bar")
    st.pyplot(fig_2)


    #fig_3 = shap.summary_plot(shap_values, df_client_2)
    #st.pyplot(fig_3)


    st.subheader("Graphique №2")
    st.caption("Un graphique montrant le raisonnement qui pousse le modèle à classer une personne dans cette classe par rapport aux valeurs des variables les plus affectées. Les variables qui augmentent la probabilité de défaut sont en rouge, tandis que celles qui diminuent la probabilité sont en bleu.")

    fig_4 = shap.force_plot(explainer.expected_value, shap_values, df_client_2, show=False, matplotlib=True, figsize= (20,5), text_rotation=30).savefig('features.png')
    st.image('features.png')
    

    st.subheader("Importance des features")
    df_shap_test = pd.DataFrame(shap_values, columns=df_client_2.columns.values)
    df_shap_test = df_shap_test.T
    df_shap_test = df_shap_test.rename(columns={0: 'Values-Importance'})
    df_shap_test = df_shap_test.sort_values(by='Values-Importance', ascending=False)
    df_shap_test = df_shap_test.reset_index()
    df_shap_test = df_shap_test.rename(columns={'index': 'Features'})
    st.write(df_shap_test)









