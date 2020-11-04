# -*- coding: utf-8 -*-
"""
@author: Andrey Ferreira de Almeida
"""

#bibliotecas
import pandas as pd
from random import randrange as rnd
import numpy as np
from sklearn.datasets import load_diabetes

#carrega a base de dados
diabetes = load_diabetes()

#importo a base de dados
db = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

#numero de clusters
k = 3

#cria os centroides aleatórios
centroids = []

for i in range(k):
    linha = rnd(db.shape[0] + 1)
    if linha not in centroids:
        centroids.append(linha)
    else:
        linha = rnd(db.shape[0] + 1)
        if linha not in centroids:
            centroids.append(linha)
        else:
            linha = rnd(db.shape[0] + 1)
            centroids.append(linha)
       
centroids_r = db.loc[centroids, :]

#formata como uma matriz
dbc = db.copy()
dbc = dbc[:].values

#formata como uma matriz
centroids_r = centroids_r[:].values

#cria os novos clusters zerados
clusters_n = pd.Series([float(0) for x in range(db.shape[0])])

iteracao = True

while(iteracao):

    #cria a nova matriz com zeros de tamanho do dataset original e com k colunas
    db_n = [[float(0) for a in range(k)] for b in range(db.shape[0])]
    
    #calcula as distancias dos pontos
    for h in range(k):
        for i in range(db.shape[0]):
            dist = 0
            for j in range(db.shape[1]):
                dist += abs(dbc[i][j] - centroids_r[h][j])
            db_n[i][h] = dist
    
    #formata como uma matriz (número de exemplos, número de clusters)
    db_n = pd.DataFrame(db_n)
    
    #define os menores clusters
    clusters = db_n.idxmin(axis=1)
    
    #verifica se o cluster se repetiu
    comparacao = clusters.eq(clusters_n)
    
    #caso seja igual não continua a iteracao porque chegou no critério de parada
    resultado = all(comparacao)
    
    if resultado:
        iteracao = False
    #senão, atribui o valor do cluster atual no antigo
    else:
        clusters_n = clusters
    
    #recalcula os centroides
    db_f = db.copy()
    
    #base final com os clusters
    db_f['clusters'] = clusters
    
    for i in range(k):
        base = db_f[db_f['clusters'] == i]
        base = base.loc[:, ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']]
        base = base[:].values
        
        medias = pd.DataFrame(np.nan_to_num(base.mean(axis=0)))[:].values.T
        
        for j in range(base.shape[1]):
            centroids_r[i][j] = medias[0][j]
