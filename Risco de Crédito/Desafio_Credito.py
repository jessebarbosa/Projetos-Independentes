#!/usr/bin/env python
# coding: utf-8

# In[367]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[368]:


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


# # Importando dados

# In[377]:


df = pd.read_csv('desafio_de_credito.csv', index_col = 'id')
df.columns


# # Limpando a Base de dados

# In[378]:


df = df.loc[df.default.notna(), :]
df.drop('Unnamed: 27', axis=1, inplace=True)
df.head()


# In[379]:


def change_notation(s):
    if isinstance(s, float):
        return s
    s = s.replace('E+',' --- ')
    lst = s.split(' --- ')
    if len(lst) == 2:
        return float(lst[0])*int(lst[1])
    else:
        return np.nan

lst_to_change = ['var_f', 'borrowed', 'income', 'ok_since']
for feature in lst_to_change:
    df[feature] = df[feature].apply(lambda s: change_notation(s))
df[['var_f', 'borrowed', 'income', 'ok_since']].head()


# In[380]:


df_res_1 = pd.concat([df.dtypes,df.isna().sum()], axis=1)
df_res_1.columns = ['Dtype', 'NaNs']
df_res_1 = df_res_1[df_res_1['NaNs'] !=0 ]
df_res_1


# In[381]:


cols_miss_obj = ['reason', 'sex', 'sign', 'social_network']
for col in cols_miss_obj:
    df.loc[:,col] = df.loc[:,col].fillna(method = 'bfill')
    
df['job_name'] = df['job_name'].fillna('nao identificado')

cols_miss_num = ['var_f', 'borrowed', 'limit', 'income', 'ok_since', 'n_bankruptcies',
                 'n_defaulted_loans', 'n_issues']

for col in cols_miss_num:
    df.loc[:,col] = df.loc[:,col].apply(lambda s: float(s) if isinstance(s, float) else np.nan).fillna(df.loc[:,col].median())


# In[382]:


df_res_2 = pd.concat([df.dtypes,df.isna().sum()], axis=1)
df_res_2.columns = ['Dtype', 'NaNs']
df_res_2.loc[list(df_res_1.index),:]


# In[383]:


df_res_3 = pd.concat([df.dtypes,df.nunique()], axis=1)
df_res_3.columns = ['Dtype', 'n_classes']
df_res_3


# # Seleção de variáveis
# Variáveis categoricas com numero muito grande de classes tornariam o modelo inviável, selecionamos então apenas as variáveis categoricas com n_classes =< 10

# In[384]:


df.drop(['var_e', 'reason', 'sign', 'state', 'zipcode', 'job_name'], axis=1, inplace=True)


# # Quantidade de individos com e sem dívida ativa

# In[385]:


sns.countplot(df.default)
plt.title('Qtd. Dívida ativa')
plt.show()


# # Analisando Exploratória de variáveis categóricas

# In[386]:


def gera_barra(df, feature):
    df_grouped = df.groupby([feature, 'default']).risk.count()
    df_pivot = pd.pivot_table(values = 'risk', index = feature, columns = ['default'], data = df_grouped.reset_index())
    df_pivot = df_pivot.reset_index()
    df_pivot.columns = list(df_pivot.columns.astype('str'))
    df_pivot['% of default'] = df_pivot.loc[:,'True']/df_pivot.iloc[:,1:].sum(axis=1)
    df_pivot.dropna(inplace=True)
    print(df_pivot)
    #df_pivot.plot(x = feature, y = ['False', 'True'], kind = 'bar')
    #plt.title('default')
    #plt.show()
    plt.figure(figsize = (4,2))
    sns.barplot(x = df_pivot[feature], y = df_pivot['% of default'] )
    #df_pivot.plot(x = feature, y = ['% of default'], kind = 'bar')
    plt.xticks(rotation=90)
    plt.show()

categorical_features = ['var_d', 'months',  'sex', 'social_network', 'channel', 'real_state']

for feature in categorical_features:
    gera_barra(df, feature)


# # Analise Exploratória de variáveis numéricas

# In[350]:


def gera_hist(df, feature):
    print(df.loc[df.default == True, feature].mean())
    print(df.loc[df.default == False, feature].mean())
    
    plt.figure(figsize = (4,2))
    sns.distplot(df.loc[df.default == True, feature])
    sns.distplot(df.loc[df.default == False, feature])
    plt.title(feature)
    
    plt.show()
numerical_features = ['risk', 'var_a', 'var_b','var_f', 'var_c', 'borrowed', 'limit',
                        'income', 'ok_since', 'n_bankruptcies', 'n_defaulted_loans',
                        'n_accounts', 'n_issues']

for feature in numerical_features:
    print(feature)
    gera_hist(df, feature)


# # Codificação de variáveis categoricas

# In[14]:


from sklearn.preprocessing import LabelEncoder

y = df.default
X_temp = df.drop('default', axis=1)

le = LabelEncoder()

categorical_features = ['var_d', 'months',  'sex', 'social_network', 'channel','real_state']

for col in categorical_features:
    le.fit(X_temp[col])
    X_temp[col] = le.transform(X_temp[col])
X_temp


# In[15]:


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
ohe.fit(X_temp[categorical_features])
X_ohe= ohe.transform(X_temp[categorical_features])
X_ohe = X_ohe.toarray()
X_ohe = pd.DataFrame(X_ohe, columns = ohe.get_feature_names(), index = X_temp.index)
X_ohe


# In[16]:


X = pd.concat([X_temp[numerical_features], X_ohe], axis=1)
X


# # Importancia de Variáveis
# 
# Analisamos a importancia de variáveis para excluir variáveis pouco relevantes, para simplificar o modelo

# In[17]:


from sklearn.tree import DecisionTreeRegressor

# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_

# plot feature importance
plt.figure(figsize = (10,5))

sns.barplot(x = X.columns, y = importance)

plt.xticks(rotation=90)
plt.title ('Importancia de Variáveis')
plt.show()


# In[18]:


for col in X.columns:
    if 'x0' in col:
        X.drop(col, axis=1, inplace=True)
    if 'x4' in col:
        X.drop(col, axis=1, inplace=True)
    if 'x5' in col:
        X.drop(col, axis=1, inplace=True)


# In[19]:


X


# # Classificadores e Métricas
# Buscamos um classificador que minimize o risco de não pagamento, logo usamos a métrica de Recall

# In[389]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report, roc_curve, make_scorer, roc_auc_score
from numpy import random


import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), random_state = 0)

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
scoring = make_scorer(recall_score)

for model_name in models.keys():
    model = models[model_name]
    score = cross_validate(model, X,y.astype(int), cv=5, scoring = scoring)['test_score'].mean()
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
    
    
    plt.show()
    print("\nRecall Médio: {}\n".format(score))


# # Buscando hiperparametros

# In[237]:


from sklearn.model_selection import GridSearchCV

random.seed(1)

# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': list(range(20,200,10)),
    'min_samples_leaf': [1, 2, 4, 8],
    'min_samples_split': [2, 4, 8,16]
}
# Create a based model
tree = DecisionTreeClassifier(random_state = 0)

recall = make_scorer(recall_score)
# Instantiate the grid search model
clf = GridSearchCV(estimator = tree, param_grid = param_grid,
                   n_jobs = -1, verbose = 2, scoring = recall)
# Fit the grid search to the data
clf.fit(X_train, y_train)
clf.best_params_


# In[240]:


print(clf.cv_results_.keys())
recall = np.array([clf.cv_results_['split{}_test_score'.format(i)] for i in range(5)]).mean(axis=0)
recall = recall.reshape(18,4,4)
recall_max_deep = recall.mean(axis=1).mean(axis=1)
recall_samples = recall.mean(axis=0)
#tpr = pd.DataFrame(tpr, )
sns.lineplot(x = range(20, 200,10), y = recall_max_deep)
plt.title('Otimização de Hiperparametro')

plt.ylabel('Recall')
plt.xlabel('max_deep')
plt.show()
sns.heatmap(pd.DataFrame(recall_samples, index =[1,2,4,8] , columns =[2,4,8,16]))

plt.title('Otimização de Hiperparametro')
plt.ylabel('param_min_samples_leaf')
plt.xlabel('param_min_samples_split')
plt.show()


# In[241]:


from sklearn.model_selection import GridSearchCV

random.seed(1)

# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': list(range(20,200,10))
}
# Create a based model
tree = DecisionTreeClassifier(random_state = 0)
acc = make_scorer(accuracy_score)
# Instantiate the grid search model
clf = GridSearchCV(estimator = tree, param_grid = param_grid,
                   n_jobs = -1, verbose = 2, scoring = acc )
# Fit the grid search to the data
clf.fit(X_train, y_train)
clf.best_params_


# In[243]:


acc = np.array([clf.cv_results_['split{}_test_score'.format(i)] for i in range(5)]).mean(axis=0)
#tpr = tpr.reshape(18,4,4)
#tpr_max_deep = tpr.mean(axis=1).mean(axis=1)
#tpr_min_samples = tpr.mean(axis=0)
#tpr = pd.DataFrame(tpr, )
sns.lineplot(x = range(20, 200,10), y = acc)
plt.title('Otimização de Hiperparametro')

plt.ylabel('Accuracia')
plt.xlabel('max_deep')
plt.show()
#sns.heatmap(pd.DataFrame(tpr_min_samples, index =[1,2,4,8] , columns =[2,4,8,16]))

#plt.title('Otimização de Hiperparametro')
#plt.ylabel('param_min_samples_leaf')
#plt.xlabel('param_min_samples_split')
#plt.show()


# # Modelo Otimizado por Recall

# In[291]:


random.seed(1)

model = DecisionTreeClassifier(
                             max_depth= 50,
                             min_samples_leaf= 1,
                             min_samples_split= 2,
                             random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

scoring = make_scorer(recall_score)

score = cross_validate(model, X,y.astype(int), cv=5, scoring = scoring)['test_score'].mean()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
    
#print(model_name+'\n')
print('Classification Report : ' )
print(classification_report(y_test, y_pred))
print("Accuracy on test data: {:.2f}".format(accuracy_score(y_test, y_pred)))
print('Confusion Matrix')
    
cols = ['negative', 'positive']
plt.figure(figsize=(6,4)),

mtr = pd.DataFrame(confusion_matrix(y_test, y_pred), index = cols, columns = cols )
mtr = mtr.loc[['positive', 'negative'], ['positive', 'negative']]
sns.heatmap(data = mtr, annot = True, fmt='d', cmap="Blues")
    
plt.show()
print("Recall Médio: {}\n".format(score))


# # Modelo Otimizado por Acurácia

# In[292]:


random.seed(1)

model = DecisionTreeClassifier(
                             max_depth= 20,
                             min_samples_leaf= 1,
                             min_samples_split= 2,
                             random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

scoring = make_scorer(accuracy_score)

score = cross_validate(model, X,y.astype(int), cv=5, scoring = scoring)['test_score'].mean()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
    
#print(model_name+'\n')
print('Classification Report : ' )
print(classification_report(y_test, y_pred))
print("Accuracy on test data: {:.2f}".format(accuracy_score(y_test, y_pred)))
print('Confusion Matrix')
    
cols = ['negative', 'positive']
plt.figure(figsize=(6,4)),

mtr = pd.DataFrame(confusion_matrix(y_test, y_pred), index = cols, columns = cols )
mtr = mtr.loc[['positive', 'negative'], ['positive', 'negative']]
sns.heatmap(data = mtr, annot = True, fmt='d', cmap="Blues")

plt.show()
print("\nAcurácia Média: {}\n".format(score))


# # Modelo Hibrido

# In[265]:


model_acc = DecisionTreeClassifier(
                             max_depth= 20,
                             min_samples_leaf= 1,
                             min_samples_split= 2,
                             random_state = 0)
model_recall = DecisionTreeClassifier(
                             max_depth= 50,
                             min_samples_leaf= 1,
                             min_samples_split= 2,
                             random_state = 0)

model_acc.fit(X_train, y_train)
model_recall.fit(X_train, y_train)

w1, w2  = 0.9, 0.1
y_probs_acc = model_acc.predict_proba(X_test)
y_probs_recall = model_recall.predict_proba(X_test)

y_probs = w1*y_probs_recall + w2*y_probs_acc
y_pred = (y_probs > 0.5).astype(int)[:,1]

#print(model_name+'\n')
print('Classification Report : ' )
print(classification_report(y_test, y_pred))
print("Accuracy on test data: {:.2f}".format(accuracy_score(y_test, y_pred)))
print("Recall on test data: {:.2f}".format(recall_score(y_test, y_pred)))
print('Confusion Matrix')
    
cols = ['negative', 'positive']
plt.figure(figsize=(6,4)),

mtr = pd.DataFrame(confusion_matrix(y_test, y_pred), index = cols, columns = cols )
mtr = mtr.loc[['positive', 'negative'], ['positive', 'negative']]
sns.heatmap(data = mtr, annot = True, fmt='d', cmap="Blues")

plt.show()


# In[390]:


from sklearn.metrics import precision_recall_curve

weights = [(0, 1),(0.99, 0.01), (1, 0)]



y_probs_acc = model_acc.predict_proba(X_test)
y_probs_recall = model_tpr.predict_proba(X_test)

for w1, w2 in weights:
    ns_probs = [0 for _ in range(len(y_test))]
    
    y_probs = w1*y_probs_recall + w2*y_probs_acc
    y_probs = y_probs[:,1]
    #y_pred = (y_test > 0.5).astype(int)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_probs)
    # plot the precision-recall curves
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot(lr_recall, lr_precision, marker='.', label='w1 = {} e w2 = {}'.format(w1, w2))
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
# show the plot
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.legend()
plt.show()

