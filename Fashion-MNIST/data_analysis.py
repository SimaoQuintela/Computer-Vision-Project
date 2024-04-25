import numpy as np
from get_data import *


(x_train, y_train), (x_test, y_test), classes = load_data()


print(f"\nDimensão dos dados de treino:", x_train.shape)
print(f"Dimensão dos dados de teste:", x_test.shape)


print("\nQuantidade de cada peça de roupa nos dados de treino:")
labels, quantity = np.unique(y_train, return_counts=True)
for i in labels:
    print(f"{classes[i]}: {quantity[i]}")


print("\nQuantidade de cada peça de roupa nos dados de teste:")
labels, quantity = np.unique(y_test, return_counts=True)
for i in labels:
    print(f"{classes[i]}: {quantity[i]}")


print(f"\nNúmero de missing values nos dados de treino:", np.count_nonzero(np.isnan(x_train)))  #detetar missing values
print(f"Número de missing values nos dados de teste:", np.count_nonzero(np.isnan(x_test)))



unique_rows_tr, _, _ = np.unique(x_train, axis=0, return_index=True, return_counts=True)
unique_rows_tst, _, _ = np.unique(x_test, axis=0, return_index=True, return_counts=True)
print(f"\nNúmero de linhas duplicadas nos dados de treino:", x_train.shape[0] - len(unique_rows_tr))
print(f"Número de linhas duplicadas nos dados de teste:", x_test.shape[0] - len(unique_rows_tst))
