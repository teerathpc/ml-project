#!/usr/bin/env python
# coding: utf-8

# # Descriptive Stats

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')



# # Available Seaborn datasets

# In[2]:


print(sns.get_dataset_names())


# # Load Diamond Dataset

# In[4]:


diamonds = sns.read_csv('diamonds')
diamonds.head()


# # Info

# In[7]:


diamonds.info()


# # Descriptive Statistics

# In[8]:


diamonds.describe()


# In[9]:


# Separating X,Y 
Y=diamonds.price
X=diamonds.drop('price',axis=1)


# # Visualizing outlier boxplot

# In[10]:


X.boxplot(figsize=(15,15))
plt.show()


# ## <p style='color:blue;background-color:silver;padding:5px;margin:2px;text-align:center;border-color:red;border-style:dotted;border-width:5px'> Measures of Central Tendency</p>
# 
# 
# 
# 1. Mean
# 2. Median
# 3. Mode

# In[11]:


X.mode()


# In[41]:


# Median carat
cut['carat'].median()
cut['carat'].median().plot(kind='bar')


# In[13]:


# Mode
diamonds.mode()


# In[38]:


diamonds.dtypes


# In[42]:


# InteractiveShell.ast_node_interactivity='none'

c=1
plt.figure(figsize=(20,20))
for i in diamonds.columns:
    if diamonds[i].dtype=='object':
#         diamonds[i].mode()
        plt.subplot(3,3,c,label=i)
        diamonds[i].value_counts().plot(kind='bar')
        plt.legend()
        plt.tight_layout()
        c+=1

plt.show()    #     plt.subplots()


# In[43]:


# Mean carat
cut=X.groupby('cut')
cut.mean()
cut.mean().plot(kind='bar')
plt.show()


# In[20]:


X.columns


# In[21]:


X.mode


# ## <p style='color:blue;background-color:pink;padding:5px;margin:2px;text-align:center;border-color:red;border-style:dotted;border-width:5px'> Measures of Dispersion</p>
# 
# 
# 
# 1. Range
# 2. Variance
# 3. Standard Deviation

# ## 1. Skewness
# Skewness refers to a distortion or asymmetry that deviates from the symmetrical bell curve, or normal distribution, in a set of data. If the curve is shifted to the left or to the right, it is said to be skewed.
# 
# <img src='https://i.imgur.com/6SH4wau.png'>
# 
# * Less than -1 or greater than 1, then the distribution is highly skewed.
# * Between -1 and -0.5 or 0.5 and 1, then the distribution is moderately skewed.
# * Between -0.5 and 0.5, then the distribution is approx. symmetric.

# In[24]:


Y.hist()


# In[25]:


sns.displot(Y,kde=True)
plt.show()


# In[26]:


Y.skew()


# In[27]:


np.log(Y).hist()
(np.log(Y)).skew()


# In[28]:


np.log2(Y).hist()
np.log2(Y).skew()


# In[29]:


from scipy.stats import boxcox


# In[30]:


sns.displot(boxcox(Y)[0],kde=True)
pd.Series(boxcox(Y)[0]).skew()


# ## 2. Kurtosis
# 
# 
# <img src='https://editor.analyticsvidhya.com/uploads/57983kurt1.png' alt='kurt img is supposed to be here if not chage the srcc link'>
# 
# * Kurtosis tells you the height and sharpness if the central peak.
# * A normal distribution will have kurtosis value exactly 3. It is called as Mesokurtic.
# * A distribution with kurtosis <3 is called Platykurtic. Compared to a Normal distribution, the tails are shorter and central peak is lower and broader.
# * A distribution with kurtosis >3 is called as Leptokurtic, where tails are longer and central peak is taller and sharper' alt='kurt img is supposed to be here if not chage the srcc link'>
# 
# * Kurtosis tells you the height and sharpness if the central peak.
# * A normal distribution will have kurtosis value exactly 3. It is called as Mesokurtic.
# * A distribution with kurtosis <3 is called Platykurtic. Compared to a Normal distribution, the tails are shorter and central peak is lower and broader.
# * A distribution with kurtosis >3 is called as Leptokurtic, where tails are longer and central peak is taller and sharper

# In[31]:


Y.kurt()


# ## 1. Continuous to Discrete

# In[32]:


# Binning
Y.max()-Y.min()


# In[33]:


bins=[0,3000,5000,8000,50000]
ninned=pd.cut(Y,bins=bins,labels=['cheap','moderate','expensive','premium'])
ninned.value_counts().plot(kind='bar')
plt.show()


# ## 2. Encoding

# In[34]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
Y_enc=le.fit_transform(ninned)
Y_enc


# In[ ]:




