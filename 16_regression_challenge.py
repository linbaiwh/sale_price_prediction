#!/usr/bin/env python
# coding: utf-8

# # Purchase Price Prediction
# 
# We are an agency helping our customers purchase XoX (a made-up product) from various makers. Price is one of the most important things we care about. We need to estimate the price of a XoX before we recommend it to our customers. However, the estimations vary significantly with our employees' experience. 
# 
# We would like you to build a machine learning model to accurately predict the price for a future purchase and provide insights to help us explain the predicted price to our customers. Please note that neither our customers nor us have any knowledge about machine learning. A sample of our sales record is available in ../data/sales.csv.
# 
# 
# 1. Preprocess, clean, analyze and visualized the provided data. A few sentences or plots describing your approach will suffice. Any thing you would like us to know about the price?
# 
# 2. Build a machine learning model to help us determine the price for a purchase. Discuss why you choose the approaches, what alternatives you consider, and any concerns you have. How is the performance of your model?
# 
# 3. Help us understand how the price is predicted (again, a few sentences will suffice).

# -------------------

# ## Table of Content
# 
# ### 1. Prepare Data Set
# 
#    - Input Python Packages
#    - Load and Prepare Data Set
# 
# ### 2. Data Quality Assessment
# 
#    - Check Missing Values
#    - Check Duplicated Values
#    
# ### 3. Exploratory Data Analysis and Feature Engineering
# 
#    - please input
#    - please input
#    
# ### 4. Building Models
# 
#    - please input
#    - please input
#    
# ### 5. Conclusion
# 
#    - please input
#    - please input
#    
# ### 6. Reference
# 
# 

# --------------------

# ## 1. Prepare Data Set

# ### Input Python Packages

# In[1]:


import numpy as np
import pandas as pd
import datetime as dt

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go

from scipy.stats import pearsonr

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler 
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import make_scorer


# ### Load and Prepare Data Set

# In[2]:


df = pd.read_csv('sales.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# ## 2. Data Quality Assesment

# ### Check Missing Values

# In[6]:


# check and sort out the percentage of missing values on each column
(df.isnull().sum()*100/len(df)).round(2).sort_values(ascending=False)


# #### Note: We have more than 20% of missing values on *cost* and *maker* features.

# **Price** will be our predict target for modeling, and there are only 0.34% of missing values on price column, so I will use dropna to remove null values on price. 

# In[7]:


df.dropna(subset=['price'], inplace=True)


# ### Check Duplicated Values

# In[8]:


df.duplicated().sum()


# ## 3. Exploratory Data Analysis and Feature Engineering

# ### Numerical Data Analysis

# In[9]:


# Write functions tranform string values to numerical values

def cost2num(x):
    if type(x) == str: 
        x = x.strip('$').strip('k')
        return float(x)*1000
    else:
        return x

def weight2num(x):
    if type(x) == str:
        x = x.strip("Kg").replace(" ", "").split("Ton")
        return float(x[0])*1000 + float(x[1])
    else:
        return x

def height2num(x):
    if type(x) == str: 
        x = x.strip(' meters')
        return float(x)
    else:
        return x

def width2num(x):
    if type(x) == str: 
        x = x.strip(' cm')
        return float(x)
    else:
        return x

def depth2num(x):
    if type(x) == str: 
        x = x.strip(' cm')
        return float(x)
    else:
        return x

def price2num(price):
    if type(price)==str:
        price = price.strip('$').replace(',', '')
    return float(price)


# In[10]:


# Add new columns for cost, price, weight, height, width, depth and map the functions
df['cost($)'] = df['cost'].map(cost2num)
df['price($)'] = df['price'].map(price2num)
df['weight(kg)'] = df['weight'].map(weight2num)
df['height(meters)'] = df['height'].map(height2num)
df['width(cm)'] = df['width'].map(width2num)
df['depth(cm)'] = df['depth'].map(depth2num)


# In[11]:


df.head()


# In[12]:


# Plot the distribution of numerical features
sns.pairplot(df)


# #### Observations:
# 
# 1. cost, price are highly **right-skewed**. It means they have outliers on the right side of distribution. May consider doing log transform to remove the skewness. 
# 2. **Correlation**:
# 
#    - Highly correlated: width vs depth
# 
#    - Highly negative-correlated: height vs width, height vs depth
# 
#    - Reasonably correlated: price vs cost

# #### Check Pearson correlation coefficient to prove if price and cost has high correlation

# In[13]:


plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), vmin=0, vmax=1, cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap between Numerical Features')
plt.show()


# In[14]:


# plot the outliers
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
ax = sns.boxplot(data=df, palette="Set2")


# In[15]:


# Add columns for year, month
df['year'] = pd.to_datetime(df.purchase_date).dt.year
df['month'] = pd.to_datetime(df.purchase_date).dt.strftime('%b')


# #### Trend analysis on year over year prices

# In[16]:


sns.set_theme(style="dark")

# Plot each year's time series in its own facet
g = sns.relplot(
    data=df,
    x="month", y="price($)", col="year", hue="year",
    kind="line", palette="crest", linewidth=4, zorder=5,
    col_wrap=4, height=2, aspect=1.5, legend=False,
)

# Iterate over each subplot to customize further
for year, ax in g.axes_dict.items():

    # Add the title as an annotation within the plot
    ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

    # Plot every year's time series in the background
    sns.lineplot(
        data=df, x="month", y="price($)", units="year",
        estimator=None, color=".7", linewidth=1, ax=ax,
    )

# Reduce the frequency of the x axis ticks
ax.set_xticks(ax.get_xticks()[::2])

# Tweak the supporting aspects of the plot
g.set_titles("")
g.set_axis_labels("", "Price")
g.tight_layout()

plt.title('Price Over Years')


# #### Observations:
# 1. In general, year over year the  purchase prices are pretty flat .
# 2. Low peak often happened in June, and high peaks of purchase normally in month April and September. 

# In[17]:


# Plot the relationship between price and cost
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
ax = sns.regplot(x="cost($)", y="price($)", data=df)


# #### It shows a linear corelation.

# ### Categorical Data Analysis

# In[18]:


df['product_level'].value_counts()


# In[19]:


df.product_level.value_counts().plot(kind='bar' )


# **Observation**: </font> Many rows of **product_type**, **maker**, and **ingredient** contain multiple categorical values.

# In[20]:


# Create a function to count single value on product_type, maker, and ingredient columns
def single_valuecounts(column):
    return column.str.split(',').explode().value_counts()


# In[21]:


# Top 10 product_type
single_valuecounts(df['product_type']).sort_values(ascending=False).head(10)


# In[22]:


# Top 10 maker
single_valuecounts(df['maker']).sort_values(ascending=False).head(10)


# In[23]:


# Top 10 ingredient
single_valuecounts(df['ingredient']).sort_values(ascending=False).head(10)


# ## 4. Building Models

# ### Prepare Data for Models

# In[24]:


# Transform string to numerical values

class NumricalTransformer(object):
    def __init__(self):
        self.mean = 0
    
    def fit(self, X, y=None):
        df = pd.DataFrame()
        df['cost'] = X.cost.map(self.cost2num)
        df['weight'] = X.weight.map(self.weight2num)
        df['height'] = X.height.map(self.height2num)
        df['width'] = X.width.map(self.width2num)
        df['depth'] = X.depth.map(self.depth2num)
        df['volumn'] = 100 * df['height'] * df['width'] * df['depth']
        self.mean = df.mean()
        
    def transform(self, X, y=None):
        df = pd.DataFrame()
        df['cost'] = X.cost.map(self.cost2num)
        df['weight'] = X.weight.map(self.weight2num)
        df['height'] = X.height.map(self.height2num)
        df['width'] = X.width.map(self.width2num)
        df['depth'] = X.depth.map(self.depth2num)
        df['volumn'] = 100 * df['height'] * df['width'] * df['depth']
        return df.fillna(self.mean)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def cost2num(self, x):
        if type(x) == str: 
            x = x.strip('$').strip('k')
            return float(x)*1000
        else:
            return x
        
    def weight2num(self, x):
        if type(x) == str: 
            x = x.strip(' Kg').replace(' Ton ', '.')
            return float(x)
        else:
            return x
    
    def height2num(self, x):
        if type(x) == str: 
            x = x.strip(' meters')
            return float(x)
        else:
            return x
    
    def width2num(self, x):
        if type(x) == str: 
            x = x.strip(' cm')
            return float(x)
        else:
            return x
        
    def depth2num(self, x):
        if type(x) == str: 
            x = x.strip(' cm')
            return float(x)
        else:
            return x


# In[25]:


# Transform categorical values to dummy varibles
class DummyTransformer(object):
    
    def fit(self, X, y=None):
        self.keys = set(X)
    
    def transform(self, X, y=None):
        res = {}
        for key in self.keys:
            res[key] = [0]*len(X)    
        for i, item in enumerate(X):
            if item in self.keys:
                res[item][i] = 1
        return pd.DataFrame(res)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)  


# #### Split data to training & test data set and apply transformers

# In[26]:


# print(list(df.columns))
features = ['cost',
             'price($)',
             'weight',
             'purchase_date',
             'product_type',
             'product_level',
             'maker',
             'ingredient',
             'height',
             'width',
             'depth']
target = 'price($)'
features.remove(target)


# In[27]:


X, y = df[features], df[target]


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[29]:


ntf = NumricalTransformer()


# In[30]:


dtf = DummyTransformer()


# In[31]:


ntf.fit_transform(X_train)


# In[32]:


ntf.transform(X_test).head() 


# In[33]:


dtf.fit_transform(X_train)


# In[34]:


dtf.transform(X_test).head() 


# #### Apply PCA for dimensionality-reduction
# 
# It normally uses in a large data set. I don't think we need to use here(from previous homework, it didn't improve that much for model performance). However for learning purpose, I will use here.

# In[35]:


pca_test = PCA().fit(ntf.fit_transform(X_train))
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# n_components = 2 will be a good number.

# In[36]:


pca = PCA()
X_train_pca = pca.fit_transform(ntf.fit_transform(X_train))

# transfer to df
X_train_df = pd.DataFrame(X_train_pca)
X_train_df.head()


# In[37]:


pca.explained_variance_ratio_


# ### Apply Machine Learning Models

# ### Xgboost

# In[38]:


from sklearn.pipeline import Pipeline
steps = [('ntf', NumricalTransformer()),
         ('dtf', DummyTransformer()),
         #('Rescale', StandardScaler()),
         #('pca', PCA()),
         ('xgbr', XGBRegressor())]
model = Pipeline(steps)


# In[39]:


model.fit(X_train, y_train)


# In[40]:


y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)


# In[41]:


print("\033[1m" + 'Train loss'+ "\033[0m")
print('train MAE: {0:.2e}'.format(mean_absolute_error(y_train, y_train_pred)))
print('train MSE: {0:.2e}'.format(mean_squared_error(y_train, y_train_pred)))
print('train R2: {0:.3f}'.format(r2_score(y_train, y_train_pred)))

print("\n""\033[1m" + 'Test loss'+ "\033[0m")
print('test MAE: {0:.2e}'.format(mean_absolute_error(y_test, y_test_pred)))
print('test MSE: {0:.2e}'.format(mean_squared_error(y_test, y_test_pred)))
print('test R2: {0:.3f}'.format(r2_score(y_test, y_test_pred)))


# #### Plot the feature importance in a bar chart

# In[43]:


importances = model.steps[2][1].feature_importances_

for i, v in enumerate(importances):
    print('Feature: %0d, Score: %.3f' % (i, v))

plt.bar(range(len(importances)), importances, color=('r', 'c','y', 'k', 'g', 'b'), align = "center")
plt.xticks(range(len(importances)), ['cost', 'weight', 'height', 'width', 'depth', 'volumn'])


# #### Tune hyperparameters with k-fold cross validation to optimize model performance

# In[44]:


parameters = {'xgbr__gamma': [0.0, 0.1, 0.2, 0.3],
    'xgbr__learning_rate': [0.05, 0.1],
    'xgbr__n_estimators': [100, 300, 500],
    'xgbr__max_depth': [3, 5, 10],
    'xgbr__lambda': [0.5, 1, 5],
    'xgbr__min_child_weight': [3, 5, 7]}


# In[45]:


scorer = make_scorer(mean_squared_error, greater_is_better = False)


# In[46]:


steps = [('ntf', NumricalTransformer()),
         ('dtf', DummyTransformer()),
         ('Rescale', StandardScaler()),
         #('pca', PCA()),
         ('xgbr', XGBRegressor())]
model = Pipeline(steps)


# In[47]:


model_gsv = GridSearchCV(model, parameters, cv = 5, scoring = scorer)


# In[48]:


model_gsv = model_gsv.fit(X_train, y_train)


# In[49]:


model_gsv.best_params_


# In[50]:


print('train MAE: {0:.2e}'.format(mean_absolute_error(y_train, y_train_pred)))
print('train MSE: {0:.2e}'.format(mean_squared_error(y_train, y_train_pred)))
print('train R2: {0:.3f}'.format(r2_score(y_train, y_train_pred)))

print("\n""\033[1m" + 'Test loss'+ "\033[0m")
print('test MAE: {0:.2e}'.format(mean_absolute_error(y_test, y_test_pred)))
print('test MSE: {0:.2e}'.format(mean_squared_error(y_test, y_test_pred)))
print('test R2: {0:.3f}'.format(r2_score(y_test, y_test_pred)))


# ### Loop to a series models and check the performance

# In[51]:


models = [LinearRegression(), Lasso(), Ridge(alpha = 0.5), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for model_name in models:
    
    steps = [('ntf', NumricalTransformer()),
             ('dtf', DummyTransformer()),
             ('Rescale', StandardScaler()),
             #('pca', PCA()),
             ('model', model_name)]
    model = Pipeline(steps)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print("\n")
    print(model_name)
    print("\n")  
    print("\033[1m" + 'Train loss'+ "\033[0m")
    print('train MAE: {0:.2e}'.format(mean_absolute_error(y_train, y_train_pred)))
    print('train MSE: {0:.2e}'.format(mean_squared_error(y_train, y_train_pred)))
    print('train R2: {0:.3f}'.format(r2_score(y_train, y_train_pred)))
    print("\n""\033[1m" + 'Test loss'+ "\033[0m")
    print('test MAE: {0:.2e}'.format(mean_absolute_error(y_test, y_test_pred)))
    print('test MSE: {0:.2e}'.format(mean_squared_error(y_test, y_test_pred)))
    print('test R2: {0:.3f}'.format(r2_score(y_test, y_test_pred)))
    
        


# #### As for all five algorithms, I will end up taking Xgboost, since this algorithm better than all algorithms in all indicators(MAE, MSE, R2).

# ### Prices Prediction

# In[52]:


# will continue


# ## 5. Conclusion

# ## 6. Reference

# https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/
# 
# https://medium.com/@agarwal.vishal819/outlier-detection-with-boxplots-1b6757fafa21
#     

# In[ ]:




