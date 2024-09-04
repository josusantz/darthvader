import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams


def cortar_serie_temporal(df, data_inicio, data_fim):
   
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("O índice do DataFrame deve ser do tipo DatetimeIndex.")
    
    if isinstance(data_inicio, str):
        data_inicio = pd.to_datetime(data_inicio)
    if isinstance(data_fim, str):
        data_fim = pd.to_datetime(data_fim)
    
    corte = df.loc[data_inicio:data_fim]
    
    return corte


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

def remover_outliers_boxplot(df):
  num_cols = df.select_dtypes(include=[np.number]).columns
  for col in num_cols:
      Q1 = df[col].quantile(0.25)
      Q3 = df[col].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
  return df

def remove_outliers_std_deviation(df, column_name, threshold=3):
    # Calcular a média e o desvio padrão
    mean = df[column_name].mean()
    std_dev = df[column_name].std()
    
    # Definir os limites inferior e superior
    lower_bound = mean - (threshold * std_dev)
    upper_bound = mean + (threshold * std_dev)
    
    # Filtrar os dados dentro dos limites
    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    
    return filtered_df


def plot_missing_values(df):
    """ For each column with missing values plot proportion that is missing."""
    data = [(col, df[col].isnull().sum() / len(df)) 
            for col in df.columns if df[col].isnull().sum() > 0]
    col_names = ['column', 'percent_missing']
    missing_df = pd.DataFrame(data, columns=col_names).sort_values('percent_missing')
    rcParams['figure.figsize'] = (16, 8)
    missing_df.plot(kind='barh', x='column', y='percent_missing'); 
    plt.title('Percent of missing values in colummns')