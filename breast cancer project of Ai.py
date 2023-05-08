#!/usr/bin/env python
# coding: utf-8

# ## project for Ai
# 
# # Breast Cancer Classification
# 
#  ## Attribute Information:
# 
# # -  ID number 
# # - Diagnosis (M = malignant, B = benign)
# 
# ## Ten real-valued features are computed for each cell nucleus:
# 
# # - radius (mean of distances from center to points on the perimeter)
# # - texture (standard deviation of gray-scale values)
# # - perimeter
# # - area
# # - smoothness (local variation in radius lengths)
# # - compactness (perimeter^2 / area - 1.0)
# # - concavity (severity of concave portions of the contour)
# # - concave points (number of concave portions of the contour)
# # - symmetry

# In[1]:


# Importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


plt.style.use('ggplot')


# In[2]:


# Load the data
df = pd.read_csv('data.csv')


# In[3]:


df.head()


# In[4]:


# Data Preprocessing
df.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)


# In[5]:


df.diagnosis.unique()


# In[6]:


df['diagnosis'] = df['diagnosis'].apply(lambda val: 1 if val == 'M' else 0)


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


df.info()



# In[10]:


# checking for null values
df.isna().sum()   # There are no missing values in the data.


# In[11]:


# Exploratory Data Analysis (EDA)
plt.figure(figsize = (20, 15))
plotnumber = 1

for column in df:
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[12]:


# heatmap 

plt.figure(figsize = (20, 12))

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))

sns.heatmap(corr, mask = mask, linewidths = 1, annot = True, fmt = ".2f")
plt.show()



# In[13]:


# We can see that there are many columns which are very highly correlated which causes multicollinearity so we have to remove highly correlated features.

# removing highly correlated features

corr_matrix = df.corr().abs() 

mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
tri_df = corr_matrix.mask(mask)

to_drop = [x for x in tri_df.columns if any(tri_df[x] > 0.92)]

df = df.drop(to_drop, axis = 1)

print(f"The reduced dataframe has {df.shape[1]} columns.")



# In[14]:



# creating features and label 

X = df.drop('diagnosis', axis = 1)
y = df['diagnosis']


# In[15]:



# splitting data into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# In[16]:


# scaling data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#  ## Logistic Regression
# 
# 

# In[17]:


# fitting data to model

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# In[18]:


# model predictions

y_pred = log_reg.predict(X_test)



# In[19]:




# accuracy score

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(accuracy_score(y_train, log_reg.predict(X_train)))

log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))
print(log_reg_acc)



# In[20]:



# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[21]:


# classification report

print(classification_report(y_test, y_pred))


# # K Neighbors Classifier (KNN)

# In[22]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# In[23]:



# model predictions 

y_pred = knn.predict(X_test)


# In[24]:



# accuracy score

print(accuracy_score(y_train, knn.predict(X_train)))

knn_acc = accuracy_score(y_test, knn.predict(X_test))
print(knn_acc)


# In[25]:



# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[26]:




# classification report

print(classification_report(y_test, y_pred))


#  # Support Vector Classifier (SVC)

# In[27]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc = SVC()
parameters = {
    'gamma' : [0.0001, 0.001, 0.01, 0.1],
    'C' : [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}

grid_search = GridSearchCV(svc, parameters)
grid_search.fit(X_train, y_train)


# In[28]:



# best parameters

grid_search.best_params_



# In[29]:



# best accuracy 

grid_search.best_score_



# In[30]:


svc = SVC(C = 10, gamma = 0.01)
svc.fit(X_train, y_train)


# In[31]:


# model predictions 

y_pred = svc.predict(X_test)


# In[32]:




# accuracy score

print(accuracy_score(y_train, svc.predict(X_train)))

svc_acc = accuracy_score(y_test, svc.predict(X_test))
print(svc_acc)



# In[33]:



# confusion matrix

print(confusion_matrix(y_test, y_pred))





# In[34]:


# classification report

print(classification_report(y_test, y_pred))


#  # Gradient Boosting Classifier

# In[35]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

parameters = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.001, 0.1, 1, 10],
    'n_estimators': [100, 150, 180, 200]
}

grid_search_gbc = GridSearchCV(gbc, parameters, cv = 5, n_jobs = -1, verbose = 1)
grid_search_gbc.fit(X_train, y_train)



# In[36]:



# best parameters 

grid_search_gbc.best_params_


# In[37]:



# best score

grid_search_gbc.best_score_



# In[38]:



gbc = GradientBoostingClassifier(learning_rate = 1, loss = 'exponential', n_estimators = 200)
gbc.fit(X_train, y_train)



# In[39]:



y_pred = gbc.predict(X_test)



# In[40]:



# accuracy score

print(accuracy_score(y_train, gbc.predict(X_train)))

gbc_acc = accuracy_score(y_test, y_pred)
print(gbc_acc)



# In[41]:



# confusion matrix

print(confusion_matrix(y_test, y_pred))



# In[42]:





# classification report

print(classification_report(y_test, y_pred))



# # Best model for diagnosing breast cancer is "Gradient Boosting Classifier" with an accuracy of 98.8%.

# 
