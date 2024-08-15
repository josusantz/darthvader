import streamlit as st
import pandas as pd

st.title("Treine modelos de aprendizado de máquina")

uploaded_file = st.file_uploader("Quer analisar outros dados? Faça o upload!!", type="csv")

if uploaded_file is not None:
    # Carregar o novo DataFrame a partir do arquivo CSV
    df = pd.read_csv(uploaded_file)

else:
    df = pd.read_csv('dados_A215_H_2008-06-13_2024-01-01.csv', sep = ';', header= 9)

st.dataframe(df.head().style.highlight_max(axis=0))