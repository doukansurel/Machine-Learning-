#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


#2. Veri Onisleme
#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.txt')
#pd.read_csv("veriler.csv")
veriler


# In[3]:


from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1]
c


# In[4]:


from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)


# In[5]:


havaDurumu = pd.DataFrame(data=c,index = range(14),columns=["o","r","s"])
sonVeriler = pd.concat([havaDurumu,veriler.iloc[:,1:3]],axis=1)
sonVeriler = pd.concat([sonVeriler,veriler2.iloc[:,-2:]],axis=1)
sonVeriler


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(sonVeriler.iloc[:,:-1],sonVeriler.iloc[:,:-1],test_size=0.33,random_state=0)


# In[7]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


# In[8]:


import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int) , values = sonVeriler.iloc[:,:-1],axis=1)
X_l = sonVeriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonVeriler.iloc[:,-1:],X_l).fit()
print(model.summary())


# In[9]:


sonVeriler = sonVeriler.iloc[:,1:]


# In[12]:


import statsmodels.api as sm


X = np.append(arr = np.ones((14,1)).astype(int), values=sonVeriler.iloc[:,:-1], axis=1 )
X_l = sonVeriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonVeriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)




# In[ ]:




