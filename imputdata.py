import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Dados/dados_A215_H_2008-06-13_2024-01-01.csv',sep= ';', header = 9)

df['Data Medicao'] = pd.to_datetime(df['Data Medicao'])
df['Hora Medicao'] = pd.to_datetime(df['Hora Medicao'])

df['ano'] = df['Data Medicao'].dt.year
df['mes'] = df['Data Medicao'].dt.month
df['dia'] = df['Data Medicao'].dt.day


for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.').astype(float)


df['Hora Medicao'] = df['Hora Medicao'].astype(str)
df['hora'] = df['Hora Medicao'].str[-4:-2].astype(int)
df = df.drop(['Hora Medicao'],axis=1)
        
df = df.drop('Unnamed: 22', axis =1)
sem_nulos = df.dropna()

media= df.mean()
media_geral = df.fillna(media)

media_mes = df.groupby('mes').transform('mean')
media_mes = df.fillna(media_mes)

df['dia_mes'] = df['Data Medicao'].dt.strftime('%m-%d')
media_dia_mes = df.groupby('dia_mes').transform('mean')
media_dia = df.fillna(media_dia_mes)
media_dia = media_dia.drop(columns=['dia_mes'])
df = df.drop(columns=['dia_mes'])

def encontrar_estacao(row):
    mes_dia = (row['mes'], row['dia'])
    if mes_dia >= (3, 21) and mes_dia <= (6, 20):
        return 'outono'
    elif mes_dia >= (6, 21) and mes_dia <= (9, 22):
        return 'inverno'
    elif mes_dia >= (9, 23) and mes_dia <= (12, 20):
        return 'primavera'
    elif mes_dia >= (12, 21) or mes_dia <= (3, 20):
        return 'verao'
    return None

df['estacao'] = df.apply(encontrar_estacao, axis=1)
media_mes['estacao'] = media_mes.apply(encontrar_estacao, axis=1)
media_dia['estacao'] = media_dia.apply(encontrar_estacao, axis=1)
sem_nulos['estacao'] = sem_nulos.apply(encontrar_estacao, axis=1)
estacoes = ['primavera', 'verao', 'outono', 'inverno']

media_estacao = df.groupby('estacao').transform('mean')
media_estacao = df.fillna(media_estacao)
'''
import seaborn as sns

# Load the four datasets into separate DataFrames
# Import the pandas library
import pandas as pd

# Define the four datasets
df1 = sem_nulos
df2 = media_estacao
df3 = media_mes
df4 = media_dia

# Concatenate the four DataFrames into a single DataFrame
df = pd.concat([df1, df2, df3, df4], keys=['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'])

# Create a pairplot using the concatenated DataFrame
sns.pairplot(data=df, diag_kind='kde')

# Create a pairplot with labels and title
sns.pairplot(data=df, diag_kind='kde',
             labels=['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'],
             title='Comparison of Four Datasets')
'''
plt.figure(figsize = (20,10))
sns.scatterplot(x='Data Medicao',y="RADIACAO GLOBAL(Kj/m²)", data= sem_nulos)
plt.savefig('radiacao.png')
plt.show()

plt.figure(figsize = (20,10))
sns.scatterplot(x='Data Medicao',y="RADIACAO GLOBAL(Kj/m²)", data= media_estacao)
plt.savefig('radiacao.png')
plt.show()


plt.figure(figsize = (20,10))
sns.scatterplot(x='Data Medicao',y="RADIACAO GLOBAL(Kj/m²)", data= media_mes)
plt.savefig('radiacao.png')
plt.show()

plt.figure(figsize = (20,10))
sns.scatterplot(x='Data Medicao',y="RADIACAO GLOBAL(Kj/m²)", data= media_dia)
plt.savefig('radiacao.png')
plt.show()
