#!/usr/bin/env python
# coding: utf-8

# In[1]:


# İmport library(kütüphane eklentisi)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.Upload Data(Veri yüklemesi)
df = pd.read_csv("veri.txt")


# In[2]:


#2.1Veri ön işleme
df.head()


# In[3]:


df.describe()


# In[4]:


#Missing Values(Eksik Veriler)
missValues = pd.read_csv("missing_values.txt")


# In[5]:


missValues.head()


# In[6]:


missValues.info()


# In[7]:


#Completing missing data(Eksik verilerin tamamlanması)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
yas = missValues.iloc[:,1:4].values
print(yas)
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])


# In[8]:


yas


# In[9]:


missValues


# In[10]:


#Categorization(Kategorize etme işlemi)
#Kategorik -> Numeric
ulke = missValues.iloc[:,0:1].values
print(ulke)


# In[11]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(missValues.iloc[:,0])

print(ulke)


# In[12]:


ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


# In[13]:


#Consolidation of Data(Verilerin birleştirilmesi)
#Numpy dizilerinden -> Dataframe oluşturma
result = pd.DataFrame(data=ulke, index = range(22),columns =["fr","tr","us"])


# In[14]:


result


# In[15]:


result2 = pd.DataFrame(data = yas , index = range(22),columns=["boy","kilo","yas"])


# In[16]:


result2


# In[17]:


cinsiyet = missValues.iloc[:,-1].values


# In[18]:


result3 = pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
result3


# In[19]:


finalResult = pd.concat([result,result2,result3],axis=1)


# In[20]:


finalResult


# In[23]:


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# In[ ]:




