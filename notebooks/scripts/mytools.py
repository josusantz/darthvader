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

def IQR_Outliers (X, features):

    print('# of features: ', len(features))
    print('Features: ', features)

    indices = [x for x in X.index]
    #print(indices)
    print('Number of samples: ', len(indices))
    
    out_indexlist = []
    listoutliers = []
        
    for col in features:
       
        #Using nanpercentile instead of percentile because of nan values
        Q1 = np.nanpercentile(X[col], 25.0)
        Q3 = np.nanpercentile(X[col], 75.0)
        
        cut_off = (Q3 - Q1) * 1.5
        upper, lower = Q3 + cut_off, Q1 - cut_off
        print ('\nFeature: ', col)
        print ('Upper and Lower limits: ', upper, lower)
                
        outliers_index = X[col][(X[col] < lower) | (X[col] > upper)].index.tolist()
        outliers = X[col][(X[col] < lower) | (X[col] > upper)].values
        print('Number of outliers: ', len(outliers))
        print('Outliers Index: ', outliers_index)
        print('Outliers: ', outliers)
        listoutliers.append(len(outliers))
        out_indexlist.extend(outliers_index)
        
    #using set to remove duplicates
    out_indexlist = list(set(out_indexlist))
    out_indexlist.sort()
    print('\nNumber of rows with outliers: ', len(out_indexlist))
    print('List of rows with outliers: ', out_indexlist)
    number_outliers = len(outliers)
    number_row = len(out_indexlist)
    # outliers_dict = {features:number_outliers}
    return features, listoutliers

    
def CustomSampler_IQR (X, y):
    
    features = X.columns
    df = X.copy()
    df['Outcome'] = y
    
    indices = [x for x in df.index]    
    out_indexlist = []
        
    for col in features:
       
        #Using nanpercentile instead of percentile because of nan values
        Q1 = np.nanpercentile(df[col], 25.)
        Q3 = np.nanpercentile(df[col], 75.)
        
        cut_off = (Q3 - Q1) * 1.5
        upper, lower = Q3 + cut_off, Q1 - cut_off
                
        outliers_index = df[col][(df[col] < lower) | (df[col] > upper)].index.tolist()
        outliers = df[col][(df[col] < lower) | (df[col] > upper)].values        
        out_indexlist.extend(outliers_index)
        
    #using set to remove duplicates
    out_indexlist = list(set(out_indexlist))
    
    clean_data = np.setdiff1d(indices,out_indexlist)

    return X.loc[clean_data], y.loc[clean_data]
#gpt gereted
'''
import numpy as np

def IQR_Outliers(X, features):
    print('# of features: ', len(features))
    print('Features: ', features)

    indices = X.index.tolist()
    print('Number of samples: ', len(indices))
    
    out_indexlist = []
        
    for col in features:
        # Using nanpercentile to handle NaN values
        Q1 = np.nanpercentile(X[col], 25.)
        Q3 = np.nanpercentile(X[col], 75.)
        
        # Calculate the IQR range
        IQR = Q3 - Q1
        cut_off = IQR * 1.5
        upper, lower = Q3 + cut_off, Q1 - cut_off
        
        print('\nFeature: ', col)
        print('Upper and Lower limits: ', upper, lower)
        
        # Find outliers
        outliers = X[col][(X[col] < lower) | (X[col] > upper)]
        outliers_index = outliers.index.tolist()
        
        print('Number of outliers: ', len(outliers))
        print('Outliers Index: ', outliers_index)
        print('Outliers: ', outliers.values)
        
        # Collect outlier indices
        out_indexlist.extend(outliers_index)
    
    # Remove duplicates by converting to a set and back to list
    out_indexlist = sorted(set(out_indexlist))
    
    print('\nNumber of rows with outliers: ', len(out_indexlist))
    print('List of rows with outliers: ', out_indexlist)
    
    return out_indexlist  # Optionally return the list of outlier indices

'''

'''
from sklearn import pipeline
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory

class Pipeline(pipeline.Pipeline):

    def _fit(self, X, y=None, **fit_params_steps):
        self.steps = list(self.steps)
        self._validate_steps()
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(pipeline._fit_transform_one)

        for (step_idx, name, transformer) in self._iter(
            with_final=False, filter_passthrough=False
        ):
                        
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            try:
                # joblib >= 0.12
                mem = memory.location
            except AttributeError:
                mem = memory.cachedir
            finally:
                cloned_transformer = clone(transformer) if mem else transformer

            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
            
            if isinstance(X, tuple):    ###### unpack X if is tuple: X = (X,y)
                X, y = X
            
            self.steps[step_idx] = (name, fitted_transformer)
        
        return X, y
    
    def fit(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        
        if isinstance(Xt, tuple):    ###### unpack X if is tuple: X = (X,y)
            Xt, y = Xt 
        
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self
        '''