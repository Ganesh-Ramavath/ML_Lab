#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# In[4]:


iris=load_iris()
iris


# In[7]:


flower=iris.data
flower


# In[30]:


target_var=iris.target
target_var


# In[17]:


flower.dtype


# In[ ]:





# In[18]:


flower.shape


# In[23]:


iris.feature_names


# In[32]:


plt.scatter(flower[target_var == 0, 0], flower[target_var== 0, 1], label='Setosa', c='red')
plt.scatter(flower[target_var == 1, 0], flower[target_var== 1, 1], label='Versicolor', c='blue')
plt.scatter(flower[target_var == 2, 0], flower[target_var== 2, 1], label='Virginica', c='green')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('setosa length & width')
plt.legend()
plt.show()


# In[38]:


plt.scatter(flower[target_var == 0, 1], flower[target_var== 0, 1], label='Setosa', c='red')
plt.scatter(flower[target_var == 1, 1], flower[target_var== 1, 1], label='Versicolor', c='blue')
plt.scatter(flower[target_var == 2, 1], flower[target_var== 2, 1], label='Virginica', c='green')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('setosa length & width')
plt.legend()
plt.show()

