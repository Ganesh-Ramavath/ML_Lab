#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


np.__version__


# In[6]:


#1-D
array1=np.array([12,3,2,4,5,6])
array1


# In[7]:


array2=np.array([[11,33,45,6],[55,76,87,3]])
array2


# In[8]:


array3=np.array([[[1,2,3,4],[3,4,5,6],[8,7,9,0]]])
array3


# In[10]:


array4=((3,4,5,6))
array4


# In[11]:


print(array1.ndim)


# In[12]:


print(array2.ndim)
print(array3.ndim)
print(array4.ndim)


# In[14]:


array5=np.array([1,2,3,4,5],ndmin=5)
array5


# In[15]:


array3


# In[18]:


array3[0][2]


# In[21]:


array2.shape


# In[22]:


for i in range(0,2):
    for j in range(0,3):
        print(array2[i][j])


# In[23]:


array3.dtype


# In[25]:


array6=np.array([22,4,3,5,6],dtype='S')
array6


# In[26]:


array6con=array6.astype('i')
array6con


# In[27]:


array6con=array6.astype('f')
array6con


# In[29]:


numone=np.ones((2,2))
numone


# In[30]:


numzeros=np.zeros((3,3))
numzeros


# In[32]:


numeye=np.eye(4)
numeye


# In[36]:


range_array=np.arange(0,12,4)
range_array


# In[38]:


x=np.random.randint(low=5,size=4)
x


# In[40]:


y=np.random.randint(low=12,high=44,size=4)
y


# In[42]:


np.concatenate([array1,array4])


# In[44]:


flatarr=array3.flatten()
flatarr


# In[46]:


np.reshape(flatarr[:4],(2,2))


# In[48]:


import pandas as pd
df=pd.DataFrame(array2)
df

