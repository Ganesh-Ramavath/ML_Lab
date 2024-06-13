#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score


# In[5]:


df=pd.read_csv("weight_height_dataset.csv")
df


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.columns


# In[11]:


df.head()


# In[12]:


df.tail()


# In[13]:


df.isnull()


# In[15]:


df.isnull().sum()


# In[17]:


dff=df
print(dff)
dff.drop(['Class'],axis=1)


# In[18]:


dff.drop([0],axis=0)


# In[19]:


df.iloc[:,:-1].values


# In[20]:


df.iloc[:,:-1]


# In[46]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[47]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=6)


# In[48]:


x_train


# In[49]:


x_train.shape


# In[50]:


x_test.shape


# In[51]:


y_train.shape


# In[52]:


x_test


# In[53]:


y_test


# In[54]:


y_train


# In[55]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_trainn=sc.fit_transform(x_train)
x_testt=sc.fit_transform(x_test)
x_trainn


# In[56]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_trainn, y_train)


# In[57]:


print(classifier.predict(sc.transform([[150,46]])))


# In[58]:


y_pred=classifier.predict(x_testt)
y_pred


# In[62]:


from sklearn.metrics import confusion_matrix
cf=confusion_matrix(y_test,y_pred)
cf


# In[63]:


labels=classifier.classes_
labels


# In[65]:


ax=plt.axes()
df_cm=cf

sns.heatmap(df_cm,annot=True,annot_kws={"size":30},fmt='d',cmap="Blues",xticklabels=labels,yticklabels=labels,ax=ax)
ax.set_title('confusion matrix')
plt.show()


# In[66]:


from sklearn.metrics import classification_report,roc_curve,auc
print(classification_report(y_test,y_pred,target_names=labels))

