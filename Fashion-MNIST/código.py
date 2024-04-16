import pandas as pd
import numpy as np

dados_treino= pd.read_csv("datasets/fashion-mnist_train.csv")
dados_teste=pd.read_csv("datasets/fashion-mnist_test.csv")
#print(dados_treino)
#print(dados_teste)


print(f"dimensao dados treino:", dados_treino.shape)
print(f"dimensao dados teste:", dados_teste.shape)


print(dados_treino.iloc[:, 0].value_counts()) #há 6000 de cada peça de roupa nos dados de treino
print(dados_teste.iloc[:, 0].value_counts()) #1000 nos de teste

print(f"número de missing values nos dados de treino:", dados_treino.isnull().sum().sum()) #detetar missing values
print(f"número de missing values nos dados de teste:", dados_teste.isnull().sum().sum())