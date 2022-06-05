#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation 
# ## Task 1 : Prediction using Supervised ML
# ### To predict the percentage of marks of the students based on the number of hours they studied. 

# ### Author : Priyam Mahajan 

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[ ]:


# Reading the Data 
df = pd.read_csv('http://bit.ly/w-data')
data.head(5)


# In[37]:


# Checking if data is null 
df.info()
df.isnull == True


# ### There are no null values in the dataset. 

# In[38]:


#Plotting the dataset. 

plt.scatter(df['Scores'], df['Hours'])


# In[39]:


#Checking if correlation is postive or negative. 
#Plotting graph 
sns.regplot(x= df['Hours'], y= df['Scores'])
plt.title('Regression Plot')
plt.ylabel('Marks Percentage')
plt.xlabel('Hours Studied')
plt.show()
print(data.corr())


# ### Positive correlation between variables confirmed. 

# ### Training the Model

# In[46]:


# 1. Splitting the Data 
X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2,random_state=0)


# In[56]:


#Double checking the size of test and train data. 
len(X_train)
len(X_test)


# In[58]:


X_train


# In[66]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()


# In[82]:


#Fitting the model 

reg.fit(X_train, y_train)
print('Training complete ! ')


# In[84]:


#making predictions based on the model.
pred_y = reg.predict(X_test)


# In[89]:


#Comparing actual vs predicted data. 

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': pred_y})  
df1


# In[101]:


# Plotting actual vs predicted data
plt.scatter(x = X_test, y = y_test, color = 'blue')
plt.plot(X_test, pred_y)
plt.title('Actual vs Predicted')
plt.ylabel('Marks Percentage')
plt.xlabel('Hours Studied')
plt.show()


# ### Evaluating the model. 

# In[110]:


#Checking accuracy score of model 

reg.score(X_test, y_test)


# In[103]:


#Checking accuracy of model 
from sklearn.metrics import mean_absolute_error

print('Mean absolute error: ',mean_absolute_error(y_test,pred_y))


#  ### Predicting the score of a student who has studied for 9.25 hours/day

# In[108]:


hours = [9.25]
req_pred = reg.predict([hours])
print("Predicted Score = {}".format(round(req_pred[0],3)))


# ### Therefore according to this model if a student studies for 9.25 hours/day, they are likely tos core 93.692 % marks.
