import pandas as pd
import numpy as np

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

def remover_outliers(df):
  num_cols = df.select_dtypes(include=[np.number]).columns
  for col in num_cols:
      Q1 = df[col].quantile(0.25)
      Q3 = df[col].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
  return df

