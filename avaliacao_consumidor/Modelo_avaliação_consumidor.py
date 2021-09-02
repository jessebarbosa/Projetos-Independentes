#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all the tools
import tensorflow as tf
import tensorflow.keras as keras
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


# # Importando dados do Web Scraping

# In[37]:


df = pd.read_csv('outputs/dados_avaliacão_consumidor_total.csv')

df


# In[45]:


# distribuição das avaliações

sns.countplot(df.Rating)
plt.show()
df.groupby('Rating').count()


# # Pré processamento dos textos

# In[41]:


# remove simbolos e torna maiusculo em minusculo
def format_text(df,col):
  #Remove @ tags
  comp_df = df.copy()
    
  # remove all the punctuation
  comp_df[col] = comp_df[col].str.replace(r'(@\w*)','')

  #Remove URL
  comp_df[col] = comp_df[col].str.replace(r"http\S+", "")

  #Remove # tag and the following words
  comp_df[col] = comp_df[col].str.replace(r'#\w+',"")

  #Remove all non-character
  comp_df[col] = comp_df[col].str.replace(r"[^a-zA-Z ]","")

  # Remove extra space
  comp_df[col] = comp_df[col].str.replace(r'( +)'," ")
  comp_df[col] = comp_df[col].str.strip()

  # Change to lowercase
  comp_df[col] = comp_df[col].str.lower()

  return comp_df

formated_df = format_text(df,'Review')

# Drop the columns we don't want
#formated_comp_df.drop(['id','date','flag','user'],axis=1,inplace=True)

formated_df.head()


# # Remove avaliações em outros idiomas

# In[42]:


no_pt_es = stopwords.words('spanish')
no_pt_en = stopwords.words('english')
no_pt_it = stopwords.words('italian')
no_pt = np.concatenate([no_pt_es, no_pt_en, no_pt_it])
no_pt = [word for word in no_pt if not word in stopwords.words('portuguese')]


# In[58]:


df_pt = formated_df.copy()
df_pt['Review'] = df_pt['Review'].apply(lambda s: s.split(' '))
for word in no_pt:
    index_pt = [idx for idx in df_pt.index if not word in df_pt.loc[idx, 'Review']]
    df_pt = df_pt.loc[index_pt,:]
    print(word)


# # Simplificando palavras
# Usamos o radical no lugar da própria palavra, assim diminuimos o número de palavras no modelo

# In[59]:


from nltk import PorterStemmer, word_tokenize
stemmer = PorterStemmer()

df_pt['Review'] = df_pt['Review'].apply(lambda s: ' '.join(s))

df_pt['Review'] = df_pt['Review'].apply(lambda s: word_tokenize(s))
df_pt['Review'] = df_pt['Review'].apply(lambda lst: [stemmer.stem(i) for i in lst])
df_pt['Review'] = df_pt['Review'].apply(lambda lst: ' '.join(lst))

df_pt


# # Palavras mais frequentes

# In[60]:


def show_wc(df,stopword=False):
    if stopword:
        stop_words = stopwords.words('portuguese')
        wc = WordCloud(max_words=3000, background_color='white',stopwords=stop_words,colormap='rainbow',height=500,width=1000)
    else:
        wc = WordCloud(max_words=3000, background_color='white',colormap='rainbow',height=500,width=1000)
    text =df.Review.values
    wc.generate(str(text))
    
    fig = plt.figure()
    plt.imshow(wc)
    fig.set_figwidth(10)
    fig.set_figheight(10)

    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# In[61]:


show_wc(df_pt, stopword=True)


# # Agrupando em classes
# 
# Devido a falta de dados sobre as avaliações com rating menor que 5, agrupamos os textos em duas classes:
#  - positivas (que recebem avaliações = 5)
#  - negativas (que recebem avaliações < 5) 

# In[62]:


# show the distribution of labels
sns.countplot(df_pt.Rating)
plt.show()
df_pt.groupby('Rating').count()


# In[56]:


df_pt['Sentiment'] = df_pt['Rating'].apply(lambda s: 'positive' if s==5 else 'negative')

sns.countplot(df_pt.Sentiment)
plt.show()
df_pt.groupby('Sentiment').count().Review


# In[103]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(df_pt['Review'].values).toarray()


# In[128]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, make_scorer, roc_auc_score
from numpy import random

import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

random.seed(1)

models = {
            "Arvore de Decisão": DecisionTreeClassifier(random_state = 0),
            "Floresta Aleatória": RandomForestClassifier(random_state = 0),
            "Regressão Logistica": LogisticRegression(solver='liblinear',random_state = 0),
            "GradientBoosting": GradientBoostingClassifier(random_state = 0),
            "Suport Vector Classifier": SVC(random_state = 0),
            "Naives Bayes Classifier": GaussianNB()
}
score_lst = []
#scoring = {
#           'TPR': make_scorer(true_positive_rate)}

for model_name in models.keys():
    model = models[model_name]
    tpr = cross_validate(model, X,y, cv=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(model_name+'\n')
    print('Classification Report : ' )
    print(classification_report(y_test, y_pred))
    print("Accuracy on test data: {:.2f}".format(accuracy_score(y_test, y_pred)))
    print('Confusion Matrix')
    
    cols = ['negative', 'positive']
    plt.figure(figsize=(6,4)),

    mtr = pd.DataFrame(confusion_matrix(y_test, y_pred), index = cols, columns = cols )
    mtr = mtr.loc[['positive', 'negative'], ['positive', 'negative']]
    sns.heatmap(data = mtr, annot = True, fmt='d', cmap="Blues")
    tpr = mtr.loc['positive', 'positive']/(mtr.loc['positive', 'positive'] + mtr.loc['positive', 'negative'])
    
    plt.show()
    print("Acuraccia média: {}\n".format(tpr))


# In[118]:


from sklearn.model_selection import GridSearchCV

random.seed(1)

# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': list(range(20,200,10)),
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 4, 8]
}
# Create a based model
GB = GradientBoostingClassifier(random_state = 0)

# Instantiate the grid search model
clf = GridSearchCV(estimator = tree, param_grid = param_grid,
                   n_jobs = -1, verbose = 2)
# Fit the grid search to the data
clf.fit(X_train, y_train)
clf.best_params_


# In[120]:


print(clf.cv_results_.keys())
tpr = np.array([clf.cv_results_['split{}_test_score'.format(i)] for i in range(5)]).mean(axis=0)
tpr = tpr.reshape(18,3,3)
tpr_max_deep = tpr.mean(axis=1).mean(axis=1)
tpr_min_samples = tpr.mean(axis=0)
#tpr = pd.DataFrame(tpr, )
sns.lineplot(x = range(20, 200,10), y = tpr_max_deep)
plt.title('Otimização de Hiperparametro')

plt.ylabel('Acurácia')
plt.xlabel('max_deep')
plt.show()
sns.heatmap(pd.DataFrame(tpr_min_samples, index =[1,2,4] , columns =[2,4,8]))

plt.title('Otimização de Hiperparametro')
plt.ylabel('param_min_samples_leaf')
plt.xlabel('param_min_samples_split')
plt.show()


# In[129]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, roc_curve, auc, classification_report, recall_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 2021)


# In[126]:


model = GradientBoostingClassifier(
                        max_depth =  20,
                        min_samples_leaf = 4,
                        min_samples_split =  8)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print('Classification Report : ' )
print(classification_report(y_test, y_pred))
print("Accuracy on test data: {:.2f}".format(accuracy_score(y_test, y_pred)))
print('Confusion Matrix')

cols = ['negative', 'positive']
plt.figure(figsize=(6,4))
mtr = pd.DataFrame(confusion_matrix(y_test, y_pred), index = cols, columns = cols )
sns.heatmap(data = mtr, annot = True, fmt='d', cmap="Blues")

