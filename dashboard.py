import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.graph_objects as go

st.title("Data Visualization Solar Project")

# DataFrame padrão (pode ser substituído depois pelo upload)
df_padrao = pd.read_csv('sem_nulos_cropped.csv')

# Verificar se a coluna 'Data_Hora' existe e convertê-la para datetime
if 'Data_Hora' in df_padrao.columns:
    df_padrao['Data_Hora'] = pd.to_datetime(df_padrao['Data_Hora'])
    df_padrao.set_index('Data_Hora', inplace=True)
else:
    st.error("A coluna 'Data_Hora' não foi encontrada no DataFrame padrão.")
    st.stop()

# Verificar se o índice está no formato datetime
if not pd.api.types.is_datetime64_any_dtype(df_padrao.index):
    st.error("O índice do DataFrame padrão não está no formato datetime.")
    st.stop()

# Opção para upload de novo arquivo CSV
uploaded_file = st.file_uploader("Quer analisar outros dados? Faça o upload!!", type="csv")

# Se um novo arquivo for carregado, substituir o DataFrame padrão
if uploaded_file is not None:
    # Carregar o novo DataFrame a partir do arquivo CSV
    df = pd.read_csv(uploaded_file)

    # Verificar se a coluna 'Data_Hora' existe e convertê-la para datetime
    if 'Data_Hora' in df.columns:
        df['Data_Hora'] = pd.to_datetime(df['Data_Hora'])
        df.set_index('Data_Hora', inplace=True)
    else:
        st.error("A coluna 'Data_Hora' não foi encontrada no DataFrame carregado.")
        st.stop()

    # Verificar se o índice está no formato datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        st.error("O índice do DataFrame carregado não está no formato datetime.")
        st.stop()

else:
    # Usar o DataFrame padrão
    df = df_padrao

# Adicionar widgets interativos
hora_min, hora_max = st.slider('Selecionar intervalo de horas do dia', 0, 23, (0, 23))
estacao_selecionada = st.multiselect('Selecionar estação do ano', options=['primavera', 'verao', 'outono', 'inverno'], default=['primavera', 'verao', 'outono', 'inverno'])

# Filtrar o DataFrame com base na seleção do usuário
df_filtrado = df[(df.index.hour >= hora_min) & (df.index.hour <= hora_max) & (df['estacao'].isin(estacao_selecionada))]

fig, axes = plt.subplots(2, 2, figsize=(18, 9))
axes = axes.flatten()

# Crie um box plot para cada estação do ano selecionada
for ax, estacao in zip(axes, estacao_selecionada):
    dados_estacao = df_filtrado[df_filtrado['estacao'] == estacao].copy()
    if not dados_estacao.empty:  # Verifique se há dados para esta estação
        dados_estacao['hora_str'] = dados_estacao.index.hour.astype(str).str.zfill(2)
        dados_estacao.boxplot(column='RADIACAO GLOBAL(Kj/m²)', by='hora_str', grid=True, ax=ax)
        ax.set_title(estacao.capitalize())
        ax.set_xlabel('Hora do Dia')
        ax.set_ylabel('Radiação Global (Kj/m²)')
        ax.get_figure().suptitle('')  # Remove o título gerado automaticamente pelo boxplot

# Ajuste o layout para evitar sobreposição
plt.tight_layout()

# Remova subplots não utilizados
for i in range(len(estacao_selecionada), len(axes)):
    fig.delaxes(axes[i])

# Exiba o box plot no Streamlit
st.pyplot(fig)

# Gráfico de médias mensais por ano usando Plotly
df['Ano'] = df.index.year
df['Mês'] = df.index.month
media_mensal_por_ano = df.groupby(['Ano', 'Mês'])['RADIACAO GLOBAL(Kj/m²)'].mean().reset_index()

# Seleção de múltiplos anos
anos_selecionados = st.multiselect('Selecionar anos para média mensal', sorted(media_mensal_por_ano['Ano'].unique()))

# Filtrar os dados para os anos selecionados
dados_anos = media_mensal_por_ano[media_mensal_por_ano['Ano'].isin(anos_selecionados)]

# Criar o gráfico Plotly
fig_plotly = go.Figure()

for ano in anos_selecionados:
    dados_ano = dados_anos[dados_anos['Ano'] == ano]
    fig_plotly.add_trace(go.Scatter(x=dados_ano['Mês'], y=dados_ano['RADIACAO GLOBAL(Kj/m²)'], mode='lines', name=str(ano)))

# Adicionando título e rótulos dos eixos
fig_plotly.update_layout(
    title='Média Mensal por Ano',
    xaxis=dict(title='Mês'),
    yaxis=dict(title='Radiação Global (Kj/m²)')
)

# Mostrando o gráfico no Streamlit
st.plotly_chart(fig_plotly)

