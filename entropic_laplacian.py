#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Entropic Laplacian Eigenmaps

Created on Wed Jul 24 16:59:33 2019

@author: Alexandre L. M. Levada

"""

# Imports
import sys
import time
import warnings
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from numpy import trace
from numpy import dot
from numpy.linalg import det
from scipy.linalg import eigh
from numpy.linalg import inv
from sklearn import preprocessing
from sklearn import metrics
import sklearn.neighbors as sknn
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
import sklearn.neighbors as sknn
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Calcula a divergência KL entre duas Gaussianas multivariadas
def divergenciaKL(mu1, mu2, cov1, cov2):
    m = len(mu1)
    
    # If covariance matrices are ill-conditioned
    if np.linalg.cond(cov1) > 1/sys.float_info.epsilon:
        cov1 = cov1 + np.diag(0.001*np.ones(m))
    if np.linalg.cond(cov2) > 1/sys.float_info.epsilon:
        cov2 = cov2 + np.diag(0.001*np.ones(m))
        
    dM1 = 0.5*(mu2-mu1).T.dot(inv(cov2)).dot(mu2-mu1)
    dM2 = 0.5*(mu1-mu2).T.dot(inv(cov1)).dot(mu1-mu2)
    dTr = 0.5*trace(dot(inv(cov1), cov2) + dot(inv(cov2), cov1))
    
    dKL = 0.5*(dTr + dM1 + dM2 - m)
    
    return dKL

# Função que implementa o Laplacian Eigenmaps
def myLaplacian(X, k, d, t, lap='padrao'):
    # Gera o grafo KNN
    knnGraph = sknn.kneighbors_graph(X, n_neighbors=k, mode='distance')
    knnGraph.data = np.exp(-(knnGraph.data**2)/t)
    W = knnGraph.toarray()  # Extrai a matriz de adjacência a partir do grafo KNN
    W = np.maximum(W, W.T)  # Para matriz de adjacência ficar simétrica

    # Matriz diagonal D e Laplaciana L
    D = np.diag(W.sum(1))   # soma as linhas
    L = D - W

    if lap == 'normalizada':
        lambdas, alphas = eigh(np.dot(inv(D), L), eigvals=(1, d))   # descarta menor autovalor (zero)
    else:
        lambdas, alphas = eigh(L, eigvals=(1, d))   # descarta menor autovalor (zero)

    return alphas

# Simple PCA implementation
def myPCA(dados, d):

    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados


# Função que implementa o método Entropic Laplacian Eigenmaps
def EntropicLaplacian(X, k, d, t, lap='padrao'):
    # Gera o grafo KNN
    knnGraph = sknn.kneighbors_graph(X, n_neighbors=k, mode='connectivity')
    W = knnGraph.toarray()  # Extrai a matriz de adjacência a partir do grafo KNN

    # Computa a média e a matriz de covariâncias para cada patch
    medias = np.zeros((X.shape[0], X.shape[1]))
    matriz_covariancias = np.zeros((X.shape[0], X.shape[1], X.shape[1]))

    for i in range(X.shape[0]):       
        vizinhos = W[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) < 2:   # pontos isolados recebem média igual a zero
            medias[i, :] = np.zeros(X.shape[1])     # pontos isolados recebem matriz de covariância igual a identidade
            medias[i, :] = amostras[i, :]
            matriz_covariancias[i, :, :] = np.eye(X.shape[1])
        else:
            amostras = X[indices]
            medias[i, :] = amostras.mean(0)
            matriz_covariancias[i, :, :] = np.cov(amostras.T)
        
    # Defines a matriz Laplaciana entrópica 
    Wkl = W.copy()
    for i in range(Wkl.shape[0]):
        for j in range(Wkl.shape[1]):
            if Wkl[i, j] > 0:
                Wkl[i, j] = divergenciaKL(medias[i, :], medias[j, :], matriz_covariancias[i, :, :], matriz_covariancias[j, :, :])

    Wkl = np.exp(-Wkl**2/t)

    # Matriz diagonal D e Laplaciana L
    D = np.diag(Wkl.sum(1))   # soma as linhas
    L = D - Wkl               # Essa é a matriz Laplaciana entrópica

    if lap == 'normalizada':
        lambdas, alphas = eigh(np.dot(inv(D), L), eigvals=(1, d))
    else:
        lambdas, alphas = eigh(L, eigvals=(1, d))

    return alphas
    

'''
 Computes the Silhouette coefficient and the supervised classification
 accuracies for several classifiers: KNN, SVM, NB, DT, QDA, MPL, GPC and RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 method: string to identify the DR method (PCA, NP-PCAKL, KPCA, ISOMAP, LLE, LAP, ...)
'''
def Classification(dados, target, method):
    print()
    print('Supervised classification for %s features' %(method))
    print()
    
    lista = []

    # 50% for training and 50% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real.T, target, test_size=.5, random_state=42)

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    acc = neigh.score(X_test, y_test)
    lista.append(acc)
    print('KNN accuracy: ', acc)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    acc = dt.score(X_test, y_test)
    lista.append(acc)
    print('DT accuracy: ', acc)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    acc = qda.score(X_test, y_test)
    lista.append(acc)
    print('QDA accuracy: ', acc)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    lista.append(acc)
    print('RFC accuracy: ', acc)

    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados.real.T, target, metric='euclidean')
    print('Silhouette coefficient: ', sc)
    
    # Computes the average accuracy
    average = sum(lista)/len(lista)
    maximo = max(lista)

    print('Average accuracy: ', average)
    print('Maximum accuracy: ', maximo)
    print()

    return [sc, average]


# Usa estratégia de busca para o melhor valor de K na construção do grafo KNN
def batch_Lap_KNN(data, target):

    n = dados.shape[0]

    # Search for the best K
    inicio = 2
    incremento = 1
    fim = min(n//2, 40)    

    vizinhos = list(range(inicio, fim, incremento))
    acuracias = []
    scs = []

    for viz in vizinhos:
        print('K = %d' %viz)
        dados_lap_ent = EntropicLaplacian(X=data, k=viz, d=2, t=1, lap='padrao') 
        s = 'LAP-KL'
        L_lapkl = Classification(dados_lap_ent.T, target, s)
        scs.append(L_lapkl[0])
        acuracias.append(L_lapkl[1])

    print('List of values for K: ', vizinhos)
    print('Supervised classification accuracies: ', acuracias)
    acuracias = np.array(acuracias)
    print('Best Acc: ', acuracias.max())
    print('K* = ', vizinhos[acuracias.argmax()])
    print()

    plt.figure(1)
    plt.plot(vizinhos, acuracias)
    plt.title('Mean accuracies for different values of K')
    plt.show()

    print('List of values for K: ', vizinhos)
    print('Silhouette Coefficients: ', scs)
    scs = np.array(scs)
    print('Best SC: ', scs.max())
    print('K* = ', vizinhos[scs.argmax()])
    print()

    plt.figure(2)
    plt.plot(vizinhos, scs, color='red')
    plt.title('Silhouette coefficients for different values of K')
    plt.show()


#%%%%%%%%%%%%%%%%%%%%  Data loading

# Scikit-learn datasets
X = skdata.fetch_openml(name='hayes-roth', version=2)          
#X = skdata.fetch_openml(name='credit-g', version=1)            
#X = skdata.fetch_openml(name='prnn_crabs', version=1)          
#X = skdata.fetch_openml(name='haberman', version=1)            
#X = skdata.fetch_openml(name='newton_hema', version=2)         
#X = skdata.fetch_openml(name='analcatdata_wildcat', version=2) 
#X = skdata.fetch_openml(name='veteran', version=2)             
#X = skdata.fetch_openml(name='datatrieve', version=1)          
#X = skdata.fetch_openml(name='grub-damage', version=2)         
#X = skdata.fetch_openml(name='disclosure_z', version=2)        
#X = skdata.fetch_openml(name='arsenic-female-bladder', version=2)
#X = skdata.fetch_openml(name='mw1', version=1)                  
#X = skdata.fetch_openml(name='ar1', version=1)                  
#X = skdata.fetch_openml(name='segment', version=2)              
#X = skdata.fetch_openml(name='kc3', version=1)                  
#X = skdata.fetch_openml(name='usp05', version=1)                
#X = skdata.fetch_openml(name='mammography', version=1)          
#X = skdata.fetch_openml(name='monks-problems-1', version=1)     
#X = skdata.fetch_openml(name='bank-marketing', version=2)       
#X = skdata.fetch_openml(name='qsar-biodeg', version=1)          
#X = skdata.fetch_openml(name='tic-tac-toe', version=1)          
#X = skdata.fetch_openml(name='pc3', version=1)                 
#X = skdata.fetch_openml(name='KnuggetChase3', version=1)       
#X = skdata.fetch_openml(name='cloud', version=2)               
#X = skdata.fetch_openml(name='blood-transfusion-service-center') 
#X = skdata.fetch_openml(name='kc1')                             
#X = skdata.fetch_openml(name='parity5', version=1)              
#X = skdata.fetch_openml(name='thoracic_surgery', version=1)     
#X = skdata.fetch_openml(name='conference_attendance', version=1)
#X = skdata.fetch_openml(name='aids', version=1)                 
#X = skdata.fetch_openml(name='fl2000', version=2)              
#X = skdata.fetch_openml(name='analcatdata_creditscore', version=1)  
#X = skdata.fetch_openml(name='analcatdata_boxing1', version=1) 
#X = skdata.fetch_openml(name='collins', version=2) 
#X = skdata.fetch_openml(name='vineyard', version=2) 
#X = skdata.fetch_openml(name='kidney', version=2)     
#X = skdata.fetch_openml(name='mux6', version=1) 
#X = skdata.fetch_openml(name='blogger', version=1)
#X = skdata.fetch_openml(name='pyrim', version=2)  
#X = skdata.fetch_openml(name='TuningSVMs', version=1)
#X = skdata.fetch_openml(name='lungcancer_GSE31210', version=1)      

dados = X['data']
target = X['target']  

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))


print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print()
input()

# Precisa tratar dados categóricos manualmente
cat_cols = dados.select_dtypes(['category']).columns
dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
# Converte para numpy (openml agora é dataframe)
dados = dados.to_numpy()
target = target.to_numpy()

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

print('############## Entropic-LAP ####################')
batch_Lap_KNN(dados, target)   # Comentar essa linha se for usar grafo KNN

#%%%%%%%%%%%% Simple PCA
dados_pca = myPCA(dados, 2)

#%%%%%%%%%%%% Kernel PCA
model = KernelPCA(n_components=2, kernel='rbf')   
dados_kpca = model.fit_transform(dados)
dados_kpca = dados_kpca.T

#%%%%%%%%%%% ISOMAP
model = Isomap(n_neighbors=20, n_components=2)
dados_isomap = model.fit_transform(dados)
dados_isomap = dados_isomap.T

#%%%%%%%%%%% LLE
model = LocallyLinearEmbedding(n_neighbors=20, n_components=2)
dados_LLE = model.fit_transform(dados)
dados_LLE = dados_LLE.T

#%%%%%%%%%%% Hessian LLE
model = LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='hessian', eigen_solver='dense')
#model = LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='hessian')
dados_LLE_h = model.fit_transform(dados)
dados_LLE_h = dados_LLE_h.T

#%%%%%%%%%%% LTSA
model = LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='ltsa', eigen_solver='dense')
#model = LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='ltsa')
dados_LLE_l = model.fit_transform(dados)
dados_LLE_l = dados_LLE_l.T

#%%%%%%%%%%% Lap. Eig.
model = SpectralEmbedding(n_neighbors=20, n_components=2)
dados_Lap = model.fit_transform(dados)
dados_Lap = dados_Lap.T

#%%%%%%%%% Classifica dados
L_pca = Classification(dados_pca, target, 'PCA')
L_kpca = Classification(dados_kpca, target, 'KPCA')
L_iso = Classification(dados_isomap, target, 'ISOMAP')
L_lle = Classification(dados_LLE, target, 'LLE')
L_lle_h = Classification(dados_LLE_h, target, 'Hessian LLE')
L_lle_l = Classification(dados_LLE_l, target, 'LTSA')
L_lap = Classification(dados_Lap, target, 'Lap. Eig.')