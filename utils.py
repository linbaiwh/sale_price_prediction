import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

def print_na(df):
    for col in df:
        if df[col].isna().any():
            print(f'{col} has {df[col].isna().sum() :.0f} missing values: {df[col].isna().sum()/df[col].isna().count() * 100 :.3f}%%')
        else:
            print(f'{col} has no missing values')

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.feature_names]

    def get_feature_names(self):
        return self.feature_names

class numeric_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = pd.DataFrame()
        if 'cost' in X.columns:
            df = df.assign(cost=X['cost'].map(lambda x: float(x.strip('$').strip('k'))*1000 if type(x)==str else x, na_action='ignore'))
        if 'weight' in X.columns:
            df = df.assign(weight=X['weight'].map(self.weight_to_num, na_action='ignore'))
        if 'height' in X.columns:
            df = df.assign(height=X['height'].map(lambda x: float(x.strip('meters'))*100 if type(x)==str else x, na_action='ignore'))
        if 'width' in X.columns:
            df = df.assign(width=X['width'].map(self.cm_to_num, na_action='ignore'))
        if 'depth' in X.columns:
            df = df.assign(depth=X['depth'].map(self.cm_to_num, na_action='ignore'))
        try:
            df = df.assign(volumn=df['height'] * df['width'] * df['depth'])
        except:
            pass
        if 'purchase_date' in X.columns:
            df = df.assign(
                year=X['purchase_date'].dt.year,
                month=X['purchase_date'].dt.month,
                weekday=X['purchase_date'].dt.weekday
            )
        
        self.df = df
        return df

    def get_feature_names(self):
        return self.df.columns.tolist()
                
    def weight_to_num(self, x):
        if type(x) == str:
            ton, kg = x.strip(' Kg').split(' Ton ')
            return float(ton) * 1000 + float(kg)
        else:
            return x
        
    def cm_to_num(self, x):
        if type(x) == str:
            return float(x.strip('cm'))
        else:
            return x

def price_to_num(price):
    return price.map(lambda x: float(x.strip('$').replace(',','')), na_action='ignore')


def plot_numeric(df, col):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(data=df, x=col, kde=True, ax=axes[0])
    axes[0].set_title(f'Histogram for {col.upper()}')

    sns.scatterplot(data=df, x=col, y='price', ax=axes[1])
    axes[1].set_title(f'Price vs. {col.upper()}')

    sns.barplot(data=df, x='year', y=col, ax=axes[2])
    axes[2].set_title(f'{col.upper()} Trend')

    fig.suptitle(f'{col.upper()} Statistics')

class freq_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq = None

    def fit(self, X, y=None):
        self.freq = X.str.split(',').explode().value_counts()
        return self

    def transform(self, X, y=None):
        freqLs = X.map(self.find_freq)

        self.df = pd.DataFrame(
            {
                # 'min_freq': freqLs.map(min, na_action='ignore'),
                'max_freq': freqLs.map(max, na_action='ignore'),
                # 'mean_freq': freqLs.map(np.mean, na_action='ignore'),
                # 'var_freq': freqLs.map(np.var, na_action='ignore'),
                'num_items': X.map(self.find_num_items)
            }
        )
        return self.df

    def get_feature_names(self):
        return self.df.columns.tolist()
    
    def find_freq(self, text):
        freqL = []
        if type(text) == str:
            items = text.split(',')
            for item in items:
                try:
                    item_freq = self.freq.loc[item]
                except KeyError:
                    pass
                else:
                    freqL.append(item_freq)
        if freqL:
            return freqL
        return None


    def find_num_items(self,text):
        if type(text) == str:
            return len(text.split(','))
        else:
            return 0 


def plot_search_results(grid):
    """Plot training/validation scores against hyperparameters

    Args:
        grid (GridSearch): GridSearch Instance that have cv_results
    """
    cv_results = pd.DataFrame(grid.cv_results_)
    # params = grid.best_params_.keys()
    params = [param[6:] for param in cv_results.columns if 'param_' in param]
    num_params = len(params)
    scores = [score[11:] for score in cv_results.columns if 'mean_train_' in score]
    num_scores = len(scores)

    fig = plt.figure(figsize=(5 * num_scores,5 * num_params))
    subfigs = fig.subfigures(num_scores, 1, squeeze=False, wspace=0.05, hspace=0.05)
#     fig, axes = plt.subplots(num_scores,num_params,squeeze=False, sharex='none', sharey='row',figsize=(5 * num_params,5 * num_scores))
    
    for j in range(num_scores):
        axes = subfigs[j,0].subplots(1, num_params, squeeze=False, sharey=True)
        subfigs[j,0].suptitle(f'{scores[j]} per Parameters')
        subfigs[j,0].supylabel('Best Score')
        
        for i, param in enumerate(params):
            param_group = cv_results.groupby(f'param_{param}')
            x = param_group.groups.keys()
            means_train = param_group[f'mean_train_{scores[j]}'].min()
            e_2 = cv_results.loc[param_group.idxmin()[f'mean_train_{scores[j]}'], f'std_train_{scores[j]}']
            means_test = param_group[f'mean_test_{scores[j]}'].min()
            e_1 = cv_results.loc[param_group.idxmin()[f'mean_test_{scores[j]}'], f'std_test_{scores[j]}']
            
            axes[0, i].errorbar(x, means_test, e_1, linestyle='--', marker='o', label='test')
            axes[0, i].errorbar(x, means_train, e_2, linestyle='-', marker='^',label='train' )
            
            axes[0, i].set_xlabel(param.upper())
            axes[0,i].legend()

#     fig.suptitle('Scores per Parameters')
#     fig.supylabel('Best Scores')
    plt.subplots_adjust(top=0.9)
    plt.legend()
    plt.show()