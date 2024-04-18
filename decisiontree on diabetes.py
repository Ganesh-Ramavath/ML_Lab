#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df=pd.read_csv("diabetes.csv")
df


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


x=df.iloc[:,:-1].to_numpy()
y=df.iloc[:,-1].to_numpy()


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[7]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion="entropy",random_state=0)
clf.fit(x_train,y_train)


# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['no','Yes'])
plt.show()


# In[9]:


clf.set_params(max_depth=3)


# In[11]:


clf.fit(x_train,y_train)
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[13]:


predicitions=clf.predict(x_test)


# In[14]:


clf.predict([[90,20],[200,30]])


# In[16]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,x_train,y_train,cv=5,scoring='accuracy')
accuracy=scores.mean()
accuracy


# In[18]:


from sklearn import metrics
cf=metrics.confusion_matrix(y_test,predicitions)
cf


# In[20]:


tp=cf[1][1]
tn=cf[0][0]
fp=cf[0][1]
fn=cf[1][0]
print(f"tp:{tp},tn:{tn},fp:{fp},fn:{fn}")


# In[21]:


print("accuracy",metrics.accuracy_score(y_test,predicitions))


# In[22]:


print("precision",metrics.precision_score(y_test,predicitions))


# In[24]:


print("Recall",metrics.recall_score(y_test,predicitions))


# In[25]:


feature_importances=clf.feature_importances_
print("feature importances:",feature_importances)


# In[ ]:




