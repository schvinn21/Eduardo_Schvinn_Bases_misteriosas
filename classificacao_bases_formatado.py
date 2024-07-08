
import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

'''
colunas_com_zeros = [col for col in df.columns if (df[col] == 0).all()]

print("Colunas com somente zeros:", colunas_com_zeros)

não foi verificada nenhuma coluna nula.

duplicatas = df.duplicated()

print("Duplicatas encontradas:")
print(duplicatas)

não foi verificado nenhuma duplicata
'''

def identify_outliers_iqr(df, factor=1.5):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    is_outlier = (df < (Q1 - factor * IQR)) | (df > (Q3 + factor * IQR))
    return is_outlier

def remove_outliers_by_threshold(df, threshold=0.15):
    outliers = identify_outliers_iqr(df)
    outliers_count = outliers.sum(axis=1)
    max_outliers = int(threshold * df.shape[1])
    df_limpa = df[outliers_count <= max_outliers]
    linhas_removidas = df[outliers_count > max_outliers]
    print("as linhas removidas foram:", linhas_removidas)
    return df_limpa

def separa_treino_teste(df_sem_outliers):
    X = df_sem_outliers.iloc[:, :-1]
    y = df_sem_outliers.iloc[:, -1]
    
    # Dividir os dados em 70% treino e 30% teste
    X_treino, X_test, y_treino, y_test = train_test_split(X, y, test_size=0.3)

    # Retornar os conjuntos de dados
    return X_treino, X_test, y_treino, y_test
    
# Normalizar os dados de treino
def normaliza_dataframe(X_treino, X_test):
    scaler = StandardScaler()
    X_treino_normalizado = pd.DataFrame(scaler.fit_transform(X_treino), columns=X_treino.columns)
    # Normalizar os dados de teste usando o mesmo scaler ajustado com base nos dados de treino
    X_test_normalizado = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_treino_normalizado, X_test_normalizado

# Selecionar as melhores características
def seleciona_melhores(X_treino_normalizado, X_test_normalizado):
    selector = SelectKBest(score_func=f_classif, k=54)
    X_treino_selecionado = selector.fit_transform(X_treino_normalizado, y_treino)
    X_test_selecionado = selector.transform(X_test_normalizado)
    # linhas, colunas = X_treino_selecionado.shape
    # print(f"Número de linhas: {linhas}, Número de colunas: {colunas}")
    return X_treino_selecionado, X_test_selecionado

def ramdom_forest_sem_tratamento(X_treino, y_treino, X_test, y_test):
    classificador = RandomForestClassifier(n_estimators=100)
    classificador.fit(X_treino, y_treino) 
    return classificador.score(X_test, y_test)

def classificador_bobo(X_treino, y_treino, X_test, y_test):
    classificador_bobo = DummyClassifier(strategy="most_frequent")
    classificador_bobo.fit(X_treino, y_treino)
    return classificador_bobo.score(X_test, y_test)

def grid_search_arvore(X_treino_selecionado, y_treino):
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': np.linspace(6, 12, 4, dtype=int),
        'min_samples_split': np.linspace(5, 20, 4, dtype=int),
        'min_samples_leaf': np.linspace(5, 20, 4, dtype=int),
        'max_features': ['sqrt', 'log2'],
        'splitter': ['best', 'random']
    }
    dt_grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=SEED),
                                  param_grid=param_grid_dt,
                                  scoring="recall",
                                  n_jobs=-1,
                                  cv=cv)
    dt_grid_search.fit(X_treino_selecionado, y_treino)
    
    return dt_grid_search.best_params_    

def grid_search_knn(X_treino_selecionado, y_treino):
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn_grid_search = GridSearchCV(estimator=KNeighborsClassifier(),
                                   param_grid=param_grid_knn,
                                   scoring='accuracy',
                                   cv=cv,
                                   verbose=1,
                                   n_jobs=-1)
    knn_grid_search.fit(X_treino_selecionado, y_treino)
    
    return knn_grid_search.best_params_ 

def grid_search_mlp(X_treino_selecionado, y_treino):
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    mlp_grid_search = GridSearchCV(estimator=MLPClassifier(max_iter=10000),
                                   param_grid=param_grid_mlp,
                                   scoring='accuracy',
                                   cv=cv,
                                   verbose=1,
                                   n_jobs=-1)
    mlp_grid_search.fit(X_treino_selecionado, y_treino)
    
    return mlp_grid_search.best_params_

# Abre o arquivo
df = pd.read_csv('12.csv')
print(df)
SEED = 20
random.seed(SEED)
# print(df)
# Remover outliers, exceto o rótulo
df_features = df.iloc[:, :-1]
df_labels = df.iloc[:, -1]

ocorrencias = df.iloc[:, -1].value_counts()
print(ocorrencias)
# Remover linhas com mais de 15% de outliers:
df_features_sem_outliers = remove_outliers_by_threshold(df_features, threshold=0.15)
df_sem_outliers = df_features_sem_outliers.join(df_labels[df_features_sem_outliers.index])
print(df_sem_outliers)
ocorrencias = df_sem_outliers.iloc[:, -1].value_counts()
print(ocorrencias)
# Separa em treino e teste:
X_treino, X_test, y_treino, y_test = separa_treino_teste(df_sem_outliers)
# print(X_treino)
'''
obtendo um base line sem tratar os dados
'''
base_line1 = ramdom_forest_sem_tratamento(X_treino, y_treino, X_test, y_test)
print(base_line1)
# base line dummy
base_line_bobo = classificador_bobo(X_treino, y_treino, X_test, y_test)
print(base_line_bobo)

# normaliza os dados:
X_treino_normalizado, X_test_normalizado = normaliza_dataframe(X_treino, X_test)

# Utilizada SelectKbest para selecionar as melhores colunas de treino e teste
X_treino_selecionado, X_test_selecionado = seleciona_melhores(X_treino_normalizado, X_test_normalizado)

'''
Após realizar o preprocessamento dos dados vamos aplicar 3 técnicas de machine learning

Algoritmo de árvore de decisão
'''

melhores_parametros_dt = grid_search_arvore(X_treino_selecionado, y_treino)
print(melhores_parametros_dt)

modelo_final_dt = DecisionTreeClassifier(**melhores_parametros_dt)
modelo_final_dt.fit(X_treino_selecionado, y_treino)
y_pred = modelo_final_dt.predict(X_test_selecionado)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

scores = cross_val_score(modelo_final_dt, X_treino_selecionado, y_treino, cv=5, scoring='accuracy')
print("Acurácia da validação cruzada: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
Algoritmo KNN
'''
melhores_parametros_knn = grid_search_knn(X_treino_selecionado, y_treino)
print(melhores_parametros_knn)
modelo_final_knn = KNeighborsClassifier(**melhores_parametros_knn)
modelo_final_knn.fit(X_treino_selecionado, y_treino)
y_pred = modelo_final_knn.predict(X_test_selecionado)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

scores_knn = cross_val_score(modelo_final_knn, X_treino_selecionado, y_treino, cv=5, scoring='accuracy')
print("Acurácia da validação cruzada: %0.2f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std() * 2))

'''
MLP
'''
# Encontrar os melhores hiperparâmetros para o MLP
melhores_parametros_mlp = grid_search_mlp(X_treino_selecionado, y_treino)
print(melhores_parametros_mlp)

# Treinar o modelo final com os melhores parâmetros
modelo_final_mlp = MLPClassifier(**melhores_parametros_mlp, max_iter=10000)
modelo_final_mlp.fit(X_treino_selecionado, y_treino)
y_pred = modelo_final_mlp.predict(X_test_selecionado)

# Imprimir o relatório de classificação e a matriz de confusão
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Avaliação com validação cruzada
scores_mlp = cross_val_score(modelo_final_mlp, X_treino_selecionado, y_treino, cv=5, scoring='accuracy')
print("\nAcurácia da validação cruzada: %0.2f (+/- %0.2f)" % (scores_mlp.mean(), scores_mlp.std() * 2))
