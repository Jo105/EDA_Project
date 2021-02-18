#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# to ignore seaborn warnings (nobody likes those)
import warnings
warnings.filterwarnings('ignore')  

plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 144)


# In[3]:


# Import the King_Country_House_prices_dataset via csv file
df = pd.read_csv('King_County_House_prices_dataset.csv')


# In[9]:


df.date = pd.to_datetime(df.date)
df.date.head(5)


# In[1]:


date_day = df.date.apply(lambda d: d.day)
date_day.name = 'date_day'

date_month = df.date.apply(lambda d: d.month)
date_month.name = 'date_month'

date_year = df.date.apply(lambda d: d.year)
date_year.name = 'date_year'

df = pd.concat([df, date_day, date_month, date_year], axis = 1)


# In[2]:


# Changing the date type for `sqft_basement`to float
# For ‘coerce’, then invalid parsing will be set as NaN
df.sqft_basement = pd.to_numeric(df.sqft_basement, errors='coerce', downcast='float')


# ## Data cleaning

# In[3]:


# Add a column to our data set to see if a house was renovated - YES '1' or NO '0'
renovated = df.yr_renovated.copy()
renovated[renovated > 0] = 1
renovated.name = 'renovated'
 
df = pd.concat([df, renovated], axis=1)


# In[22]:


# Checking the hypothesis
sum_sqft_base_sqft_above = df.sqft_basement + df.sqft_above
sqft_living = df.sqft_living
all(sqft_living[sum_sqft_base_sqft_above.notna()] == sum_sqft_base_sqft_above[sum_sqft_base_sqft_above.notna()])


# In[23]:


# Calculate the missing `sqft_basement`values and into those into our data set
df.loc[df.sqft_basement.isna(), 'sqft_basement'] =    df.sqft_living[df.sqft_basement.isna()] - df.sqft_above[df.sqft_basement.isna()]


# In[26]:


# Changeing all nan to 0 in `view`
df.loc[df.view.isna(), 'view'] = 0
df.view.isna().sum()


# ---
# # III. Exploratory Data Analysis

# In[35]:


# Get an understanding for the outlier in bedroom
df.iloc[df.bedrooms.values.argmax()].bedrooms


# In[36]:


# Get the index row for bedrooms = 33
df.index[df['bedrooms'] == 33]


# In[4]:


# add a dummy variable (1 or 0) for `basement`
basement = df.sqft_basement.copy()
basement[basement>0] = 1
basement.name = 'basement'


# In[5]:


# add the dummy basement variable to our data set
df = pd.concat([df, basement], axis=1)


# In[6]:


# create floors_dummy and add it to a new dataframe df_dummies

dummy_names = ['floor_dummy']
floors_dummy = pd.get_dummies(df.floors, prefix='floor', drop_first=True)
floors_dummy.columns = [x.replace('.', '_') for x in floors_dummy.columns]


# Since `waterfront` is a categorical variable, of course we will insert this as categorical variable in the regression model.

# In[7]:


# create waterfront_dummy and add it later to the dataframe df_dummies

dummy_names.append('waterfront_dummy')
waterfront_dummy = pd.get_dummies(df.waterfront, prefix='waterfront', drop_first=True, dummy_na = True)
waterfront_dummy.columns = ['waterfront_1', 'waterfront_nan']


# In[8]:


# create view_dummy and add it later to the dataframe df_dummies (see above why we do this)

dummy_names.append('view_dummy')
view_dummy = pd.get_dummies(df.view, prefix='view', drop_first=True)
view_dummy.columns = [x.replace('.', '_') for x in view_dummy.columns]


# In[9]:


# create renovated_dummy including a column for missing values and add it later to df_dummies

dummy_names.append('renovated_dummy')
renovated_dummy = pd.get_dummies(df.renovated, prefix='renov', drop_first=True, dummy_na = True)
renovated_dummy.columns = ['renovated_1', 'renovated_nan']


# In[67]:


# checking the relationship between renovation status (yes/no) and price per decate:

# split the `year_built`variable into decades
decades = np.arange(1900, 2030, 10)
decade_built = pd.cut(df.yr_built, bins=decades, labels= [str(d) + "s" for d in decades[:-1]])
decade_built.name = 'built_in'
df = pd.concat([df, decade_built], axis=1)


# In[69]:


# calculate mean price difference between renovated and non-renovated houses:
round(df.price[df.renovated==1].mean() - df.price[df.renovated==0].mean(),2)


# In[10]:


# create zip_dummy and add it later to the dataframe df_dummies (see above why we do this)
dummy_names.append('zip_dummy')
zip_dummy = pd.get_dummies(df.zipcode, prefix='zip', drop_first=True)


# In[83]:


# price correlation matrix with only k variables for heatmap (K-map method = Karnaugh Maps)
k = 8
cols = corrmat.nlargest(k, 'price').index   # return k columns with largest correlation with SalePries
corrmat_k = df[cols].corr()
sns.set(font_scale=1.25)
cmap = sns.cubehelix_palette(8)
plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat_k, fmt='.2f', annot=True, cmap = 'OrRd', square=True, annot_kws={'size': 12});


# In[84]:


# we should keep in mind  these numerical variables that correlate highest with price for the regression:
largest_corr = list(corrmat_k.columns[1:6])
largest_corr


# ---
# # IV. Multiple Regression Model

# In[85]:


# Step 1: select numerical variables used for regression and mean-center them

df_num = df[largest_corr].copy()
df_num = df_num.apply(lambda x: x-x.mean())
    
# df_num.drop('sqft_living', axis=1, inplace=True)
df_num.head(10)


# In[103]:


# Step 2: select categorical variables used for regression (not mean-centered)

df_cat = pd.concat([basement, waterfront_dummy, floors_dummy, view_dummy, renovated_dummy, zip_dummy], axis=1)
df_cat.head(5)


# In[104]:


# Step 3: Combine the selected numerical and categorical variables and do a multiple regression

X = pd.concat([df_num, df_cat], axis=1)
X = sm.add_constant(X)
y = df.price

sm.OLS(y, X).fit().summary()


# In[146]:


results = sm.OLS(y, X).fit()
results.save("KCH_Multiple_Regression_Model.pickle")


# # V. Modeling by using Sklearn

# In[121]:


X = pd.concat([df_num, df_cat], axis=1)

y = df.price


# In[122]:


# splitting the data set into test data (20%) and train data (80%) 

X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.2, random_state=105, shuffle=False)
print('X_Training:', X_training.shape[0],'X_Testing:', X_testing.shape[0])
print('y_Training:', y_training.shape[0],'y_Testing:', y_testing.shape[0])


# In[138]:


model_sk = LinearRegression()
model_sk.fit(X_training,y_training)


# In[139]:


y_price_prediction = model_sk.predict(X_testing)
y_price_prediction


# In[142]:


# Root mean squared error (RMSE)

print('Root Mean squared error (RMSE): %.2f'
      % mean_squared_error(y_price_prediction, y_testing, squared = False))

# r**2

print('R**2: %.2f'
      % r2_score(y_testing, y_price_prediction))


# In[ ]:




