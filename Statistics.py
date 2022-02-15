#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


data = pd.read_csv("st_data.csv")
data = data.replace(to_replace= r'/', value= '', regex=True)
data


# In[7]:


data.dtypes


# In[8]:


import seaborn as sns
sns.countplot(data['Lockdown_Type'],label="Count")
plt.show()


# In[26]:


sns.countplot(data['Fatality'],label="Count")
plt.show()


# In[10]:


data.drop('Country_Region', axis=1).plot(kind='box', subplots=True, layout=(7,7), sharex=False, sharey=False, figsize=(12,12), 
                                        title='Box Plot for each input variable')
plt.savefig('Corona')
plt.show()


# In[5]:


import pylab as pl
data.drop('Country_Region' ,axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('Corona_hist')
plt.show()


# In[6]:


from pandas.plotting import scatter_matrix
from matplotlib import cm
feature_names = ['Population_Size', 'Mean_Age', 'Latitude', 'Longtitude']
X = data[feature_names]
y = data['Lockdown_Type']
cmap = cm.get_cmap('gnuplot')
scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('fruits_scatter_matrix')


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))




from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))




from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))



from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))


# In[8]:


k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])


# In[ ]:


#Data Classification Results


# In[11]:


import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

clf = LogisticRegression().fit(X_train,y_train)
clf.predict(X_test)
y_pred = clf.predict(X_test) 

cm = confusion_matrix(y_test, y_pred) 

cm_df = pd.DataFrame(cm,)
print(cm) 
plt.figure(figsize=(6.5,4.5))
sns.heatmap(cm_df, annot=True)
plt.title('Logistic Regression \nAccuracy:{0:.4f}'.format(accuracy_score(y_test, y_pred)*100 ))
plt.ylabel('Train label')
plt.xlabel('Predicted label')
plt.show()


# In[12]:


import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

clf = DecisionTreeClassifier().fit(X_train,y_train)
clf.predict(X_test)
y_pred = clf.predict(X_test) 

cm = confusion_matrix(y_test, y_pred) 

cm_df = pd.DataFrame(cm,)
print(cm) 
plt.figure(figsize=(6.5,4.5))
sns.heatmap(cm_df, annot=True)
plt.title('Decision Tree \nAccuracy:{0:.4f}'.format(accuracy_score(y_test, y_pred)*100 ))
plt.ylabel('Train label')
plt.xlabel('Predicted label')
plt.show()


# In[13]:


import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

clf = KNeighborsClassifier().fit(X_train,y_train)
clf.predict(X_test)
y_pred = clf.predict(X_test) 

cm = confusion_matrix(y_test, y_pred) 

cm_df = pd.DataFrame(cm,)
print(cm) 
plt.figure(figsize=(6.5,4.5))
sns.heatmap(cm_df, annot=True)
plt.title('KNN \nAccuracy:{0:.4f}'.format(accuracy_score(y_test, y_pred)*100 ))
plt.ylabel('Train label')
plt.xlabel('Predicted label')
plt.show()


# In[14]:


import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

clf = SVC().fit(X_train,y_train)
clf.predict(X_test)
y_pred = clf.predict(X_test) 

cm = confusion_matrix(y_test, y_pred) 

cm_df = pd.DataFrame(cm,)
print(cm) 
plt.figure(figsize=(6.5,4.5))
sns.heatmap(cm_df, annot=True)
plt.title('Support Vector Macine \nAccuracy:{0:.4f}'.format(accuracy_score(y_test, y_pred)*100 ))
plt.ylabel('Train label')
plt.xlabel('Predicted label')
plt.show()


# In[24]:


data.corr()
sns.heatmap(data.corr(),cmap="YlGnBu",linewidths=.5)
plt.title('Corelation Heatmap')
plt.show()


# In[16]:


correlation = data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')

plt.title('Correlation between different features')


# In[8]:


data.info()


# In[ ]:


#Data Visualization and Global forecasting of corona virus


# In[6]:


# import libraries
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

df = pd.read_csv("st_data.csv")


# In[7]:


df.head()


# In[20]:


df.Lockdown_Type = df.Lockdown_Type.astype("float")
df.Fatality = df.Fatality.astype("float")


# In[23]:


import seaborn as sns
corr = df.drop(["Country_Region"],axis=1).corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,mask=mask,cmap='summer_r',vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": 0.5})


# In[9]:


temp =df[(df.Fatality == max(df.Fatality)) ]


# In[10]:


import folium
m = folium.Map(location=[0, 0], tiles='cartodbpositron',
               min_zoom=1, max_zoom=10, zoom_start=1.5)


for i in range(0, len(temp)):
    folium.Circle(
        location=[temp.iloc[i]['Latitude'], temp.iloc[i]['Longtitude']],
        color='crimson', fill='crimson',
        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country_Region'])+
                
                    '<li><bold>Deaths : '+str(temp.iloc[i]['Fatality'])+
                    '<li><bold>lockdown Type : '+str(temp.iloc[i]['Lockdown_Type'])+
                    '<li><bold>first case date : '+str(temp.iloc[i]['Date_FirstConfirmedCase'])+
                    '<li><bold>mean age : '+str(temp.iloc[i]['Mean_Age'])
        ,
        radius=int(temp.iloc[i]['Fatality']*10000)).add_to(m)

m

