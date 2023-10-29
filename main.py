import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Cargando los datos
df = pd.read_csv("datos_limpios.csv")

# Eliminando la columna categoria_edad
df = df.drop(columns=["edad_categoria"])

# Obteniendo el vector objetivo
y = df["is_dead"].values

# Dividiendo el dataset en conjunto de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["is_dead"]).values, y, stratify=y, test_size=0.25
)

# Ajustando el árbol de decisión
clf = DecisionTreeClassifier(max_depth=4, min_samples_split=50)
clf.fit(X_train, y_train)

# Calculando el accuracy sobre el conjunto de test
accuracy = clf.score(X_test, y_test)

# Imprimiendo el accuracy
print("Accuracy:", accuracy)
