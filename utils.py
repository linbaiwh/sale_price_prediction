import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm

def print_na(df):
    """Display number and percentage of missing values in each column

    Args:
        df (DataFrame): DataFrame to investigate
    """
    for col in df:
        if df[col].isna().any():
            print(f'{col} has {df[col].isna().sum() :.0f} missing values: {df[col].isna().sum()/df[col].isna().count() * 100 :.3f}%%')
        else:
            print(f'{col} has no missing values')

def print_outlier(df):
    """Display number and percentage of outliers in each column.
        Outliers are observations outside [Q1 - 1.5IQR, Q3 + 1.5IQR]

    Args:
        df (DataFrame): DataFrame to investigate
    """
    for col in df:
        IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
        lower_bound = df[col].quantile(0.25) - 1.5 * IQR
        upper_bound = df[col].quantile(0.75) + 1.5 * IQR
        num_outliers = df.loc[df[col]<lower_bound, col].count() + df.loc[df[col]>upper_bound, col].count()
        print(f'{col} has {num_outliers} outliers: {num_outliers / df[col].count() * 100 :.3f}%%')

def plot_price_dist(price):
    """Plot histogram and boxplot of price or other column, to visulize variable distribution

    Args:
        price (pandas.Series): Series to investigate
    """
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(16, 6))
    sns.histplot(price, kde=True, ax=axes[0])
    sns.boxplot(x=price, ax=axes[1])
    fig.suptitle(f'{price.name} Distribution')
        
def plot_price_trends(df):
    """Plot monthly median price and yearly median price

    Args:
        df (DataFrame): DataFrame to investigate, need to have column 'price' and 'purchase_date'
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))
    fig.supylabel('Price (LOG)')
    monthly = df.resample('M', on='purchase_date').median()
    sns.lineplot(x = 'purchase_date', y = 'price', markers=True, dashes=False, data=monthly, ax=axes[0])
    axes[0].axhline(monthly['price'].median(), color='red', linestyle='--', linewidth=2, label='Median')
    axes[0].set_title('Monthly Median Price vs Time')
    axes[0].set_xlabel('Time')
    axes[0].legend(loc='best')

    yearly = df.resample('Y', on='purchase_date').median()
    sns.lineplot(x = 'purchase_date', y = 'price', markers=True, dashes=False, data=yearly, ax=axes[1])
    axes[1].axhline(yearly['price'].mean(), color='red', linestyle='--', linewidth=2, label='Median')
    axes[1].set_title('Yearly Median Price vs Time')
    axes[1].set_xlabel('Time')
    axes[1].legend(loc='best')

    
def plot_price_seasonality(df):
    """Plot price seasonality per month, day, and weekday

    Args:
        df (DataFrame): DataFrame to investigate, need to have columns 'price' and 'purchase_date'
    """
    df['month'] = df['purchase_date'].dt.month
    df['day'] = df['purchase_date'].dt.day
    df['weekday'] = df['purchase_date'].dt.weekday

    fig, axes = plt.subplots(2, 3, figsize=(20, 6))
    sns.countplot(x="month", data=df, ax=axes[1,0])
    sns.barplot(x="month", y="price", data=df, ax=axes[0,0])
    
    sns.countplot(x="day", data=df, ax=axes[1,1])
    sns.barplot(x="day", y="price", data=df, ax=axes[0,1])
    
    sns.countplot(x="weekday", data=df, ax=axes[1,2])
    sns.barplot(x="weekday", y="price", data=df, ax=axes[0,2])

    units = ['Month', 'Day', 'Weekday']
    for i in range(len(units)):
        axes[0,i].set_title(f'Price Seasonality vs {units[i]}')
        axes[0,i].set_xlabel(units[i])

    axes[1,0].set_xticks(range(12))
    axes[1,0].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']) 
    axes[0,0].set_xticks(range(12))
    axes[0,0].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']) 

    axes[1,2].set_xticks(range(7))
    axes[1,2].set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    axes[0,2].set_xticks(range(7))
    axes[0,2].set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])

def plot_price_ar(df):
    """Plot autocorrelation and partial auto correlation for monthly median price

    Args:
        df (DataFrame): DataFrame to investigate, need to have columns 'price' and 'purchase_date'
    """
    monthly = df.resample('M', on='purchase_date').median()
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(monthly['price'].to_numpy().squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(monthly['price'], lags=40, ax=ax2)
    
    
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Sklearn transformer object to select certain columns
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.feature_names]

    def get_feature_names(self):
        return self.feature_names

class numeric_transformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer object to transform numerical data into machine readable numbers
    """
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
                day=X['purchase_date'].dt.day,
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
    """Transform price into machine readable numbers

    Args:
        price (pands.Series): column representing price in string

    Returns:
        pandas.Series: price in numbers
    """
    return price.map(lambda x: float(x.strip('$').replace(',','')), na_action='ignore')


def plot_numeric(df, col):
    """Plot histogram of a column;
        Plot scatter plot of the column and price;
        Plot bar plot of the column against month.

    Args:
        df (DataFrame): DataFrame including columns 'price', 'month', and the specified column
        col (string): name of the column to investigate
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(data=df, x=col, kde=True, ax=axes[0])
    axes[0].set_title(f'Histogram for {col.upper()}')

    sns.scatterplot(data=df, x=col, y='price', ax=axes[1])
    axes[1].set_title(f'Price vs. {col.upper()}')

    sns.barplot(data=df, x='month', y=col, ax=axes[2])
    axes[2].set_title(f'{col.upper()} Trend')

    fig.suptitle(f'{col.upper()} Statistics')

class freq_transformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer object to transform categorical data into frequency based statistics, including min_freq, max_freq, mean_freq, var_freq, and num_items.
    """
    def __init__(self, features):
        self.freq = None
        self.features = features

    def fit(self, X, y=None):
        self.freq = X.str.split(',').explode().value_counts()
        return self

    def transform(self, X, y=None):
        itemsL = X.map(self.splits_items)
        freqLs = itemsL.map(self.find_freq)

        df = pd.DataFrame(
            {
                'min_freq': freqLs.map(min, na_action='ignore'),
                'max_freq': freqLs.map(max, na_action='ignore'),
                'mean_freq': freqLs.map(np.mean, na_action='ignore'),
                'var_freq': freqLs.map(np.var, na_action='ignore'),
                'num_items': itemsL.map(len, na_action='ignore')
            }
        )
        df = df[self.features]
        return df

    def get_feature_names(self):
        return self.features
    
    def find_freq(self, items):
        """For each categorical level for one observation, find out the frequency

        Args:
            items (list of strings): list of strings representing the levels of an observation

        Returns:
            list: list of level frequencies accordingly
        """
        freqL = []
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

    def splits_items(self, text):
        if type(text) == str:
            return text.split(',')
        else:
            return []

    def plot_top_freq(self, X, y, top_count=5, **kwargs):
        """Plot boxplot for most frequent categorical levels against the target

        Args:
            X (pandas.Series): the independent variable to investigate
            y (pandas.Series): the dependent variable or the target
            top_count (int, optional): the number of most frequent levels to plot. Defaults to 5.
        """
        top_freqs = self.freq.sort_values(ascending=False)
        if len(top_freqs) > top_count:
            top_freqs = top_freqs[:top_count]
        print(f'The {top_count} most frequent items in {X.name} are:')
        print(top_freqs.to_string())
        top_freqs = set(top_freqs.index.tolist())

        top_filter = X.map(lambda x: set(self.splits_items(x)) & top_freqs, na_action='ignore')
        top_freq_X = top_filter.loc[top_filter.map(len, na_action='ignore') > 0].map(lambda x: ','.join(x))
        top_freq_y = y[y.index.isin(top_freq_X.index)]
        plt.figure(figsize=(10,4))
        sns.boxplot(x=top_freq_X, y=top_freq_y, **kwargs)
        
    def plot_freq_price(self, X, y, line_options={}, hist_options={}):
        """Plot boxplots for the transformed frequency statistics agains the target

        Args:
            X (pandas.Series): the independent variable to investigate
            y (pandas.Series): the dependent variable or the target
        """
        df = self.transform(X, y=y)
        num_plots = len(self.features)
        fig, axes = plt.subplots(2, num_plots, figsize=(7 * num_plots, 8))
        fig.suptitle(f'{X.name.upper()} Frequencies VS. Price')
        for i, col in enumerate(self.features):
            sns.lineplot(x=df[col], y=y, ax=axes[0,i], **line_options)
            sns.histplot(x=df[col], ax=axes[1,i], **hist_options)
            
class dummy_transformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer object to transform categorical variable to dummy variables;
    Allow one observation to have multiple levels
    """
    def __init__(self):
        self.keys = []
        
    def fit(self, X, y=None):
        self.keys = X.str.split(',').explode().unique().tolist()
        return self
    
    def transform(self, X, y=None):        
        res = {}
        for key in self.keys:
            res[key] = [0] * len(X)
        
        X = X.reset_index(drop=True)
        for idx, text in X.items():
            if type(text) == str:
                items = text.split(',')
                for item in items:
                    if item in self.keys:
                        res[item][idx] = 1
        
        self.df = pd.DataFrame(res)
        return self.df
    
    def get_feature_names(self):
        return self.df.columns
    
class ordinal_transformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer object to transform product levels to ordinal numbers
    """
    def __init__(self):
        self.maps = {'unrated': 0,
                    'basic': 1,
                    'intermediate': 2,
                    'advanced': 3,
                    'super advanced': 4}
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):        
        self.df = pd.DataFrame({'product_level': X.map(self.maps)})
        return self.df
    
    def get_feature_names(self):
        return self.df.columns

def plot_pca_scree(pca):
    """Plot scree plot for PCA

    Args:
        pca (sklearn.decompose.PCA): fitted sklearn PCA object
    """
    var_ratio = np.cumsum(pca.explained_variance_ratio_)
    # labels = ['PC'+str(i) for i in range(1, len(var_ratio)+1)]
    plt.plot(np.arange(1, len(var_ratio)+1), var_ratio)
    plt.xticks(rotation=90)
    plt.ylabel('Cumulative Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('Scree Plot')
    
def model_tuning(model, params, X_train, y_train):
    """Run GridSearchCV for regressions, metrics are MSE, MAE, and R2, using 3 CV groups

    Args:
        model (Pipeline): Pipeline end with a regression model
        params (dict): hyperparameters to tune
        X_train (array): train data of independent variables
        y_train (array): train data of target variable

    Returns:
        GridSearchCV: fitted GridSearchCV object
    """
    scores = {
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'R2': make_scorer(r2_score)
    }
    
    gs = GridSearchCV(model, params, n_jobs=-1, scoring=scores, cv=3, return_train_score=True, refit=False)
    gs.fit(X_train, y_train)
    return gs


def model_loss(y, y_pred, tag):
    """Display MAE, MSE, and R2

    Args:
        y (array): true target variable
        y_pred (array): predicted target variable
        tag (string): train or test
    """
    print(f'{tag} MAE: {mean_absolute_error(y, y_pred):.2e}')
    print(f'{tag} MSE: {mean_squared_error(y, y_pred):.2e}')
    print(f'{tag} R2: {r2_score(y, y_pred):.3f}')
    
    
def plot_search_results(grid):
    """Plot training/validation scores against hyperparameters

    Args:
        grid (GridSearchCV): GridSearchCV Instance that have cv_results
    """
    cv_results = pd.DataFrame(grid.cv_results_)
    # params = grid.best_params_.keys()
    params = [param[6:] for param in cv_results.columns if 'param_' in param and cv_results[param].nunique() > 1]
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

    plt.subplots_adjust(top=0.9)
    plt.legend()
    plt.show()