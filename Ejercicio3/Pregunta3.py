# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:17:30 2020

@author: Albert
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def X_Entrenamiento(X, n):
    X_T=[]
    for i in range(int(n*0.8)):#80% de la data
        X_T.append(X[i])
    X_T = np.array(X_T)
    return(X_T)
def X_Evaluacion(X, n):
    X_T=[]
    for i in range(int(n*0.8),n):#20% de la data
        X_T.append(X[i])
    X_T = np.array(X_T)
    return(X_T)
def y_Entrenamiento(y, n):
    y_T=[]
    for i in range(int(n*0.8)):#80% de la data
        y_T.append(y[i])
    return(y_T)
def y_Evaluacion(y, n):
    y_T=[]
    for i in range(int(n*0.8),n):#20% de la data
        y_T.append(y[i])
    return(y_T)


#LECTURA DEL DATASET
df = pd.read_csv("datasetP3.csv")
n = len(df.index)#Cantidad de filas dentro del dataset
#PREPROCESAMIENTO
print(df.dtypes)
"""
Se puede ver en la descripcion de las columnas obtenidas de la pagina que 6 de 
las 16 columnas son datos continuos, pero al momento de la importacion pandas
cambio el tipo de dato de las columnas A2 y A14, por ende se realizará la
conversion correspondiente.
"""
df["A2"] = pd.to_numeric(df["A2"], errors='coerce')
df["A14"] = pd.to_numeric(df["A14"], errors='coerce')
"""
Una vez realizado el proceso se tiene el resultado original de los tipos de datos
de las columnas. Con esta conversion aparecerán los datos NaN que se corregiran
más adelante.
"""
print(df.dtypes)
"""
Se procedera a asignar un valor numerico al atributo de clase del dataset asignado a la
columna A16 en el cual el valor '-' -> 1 y '+' -> 0
"""
df = pd.get_dummies(df,columns=["A16"],drop_first=True)
print(df)
#Se obtendrán las columnas numericas y de objetos en dos df diferentes
dataNum = df[["A2","A3","A8","A11","A14","A15","A16_-"]]
dataObj = df[["A1","A4","A5","A6","A7","A9","A10","A12","A13"]]
print(dataNum)
print(dataObj)
#Realizando imputacion al df numerico
imputacionN = SimpleImputer(missing_values=np.nan,strategy="mean")
dataNumImp = imputacionN.fit_transform(dataNum)
print(dataNumImp)
#Realizando imputacion al df objeto
imputacionO = SimpleImputer(missing_values="?",strategy="most_frequent")
dataObjImp = imputacionO.fit_transform(dataObj)
print(dataObjImp)
"""
Se creara un nuevo dataset a partir del nuevo dataset preprocesado y con el cual
se implementara en MLP
"""
elementos={
    "A1": dataObjImp[:,0],
    "A2": dataNumImp[:,0],
    "A3": dataNumImp[:,1],
    "A4": dataObjImp[:,1],
    "A5": dataObjImp[:,2],
    "A6": dataObjImp[:,3],
    "A7": dataObjImp[:,4],
    "A8": dataNumImp[:,2],
    "A9": dataObjImp[:,5],
    "A10": dataObjImp[:,6],
    "A11": dataNumImp[:,3],
    "A12": dataObjImp[:,7],
    "A13": dataObjImp[:,8],
    "A14": dataNumImp[:,4],
    "A15": dataNumImp[:,5],
    "A16": dataNumImp[:,6]
}
data = pd.DataFrame(elementos)
print(data)
"""
A continuacion se categorizará a los atributos de tipo objeto a tipo entero asignandole el siguiente detalle:
A1: (a, b)
    a -> 0
    b -> 1
A4: (u, y, l, t)
    u -> 0
    y -> 1
    l -> 2
    t -> 3
A5: (g, p, gg)
    g -> 0
    p -> 1
    gg -> 2
A6: (c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff)
    c -> 0
    d -> 1
    cc -> 2
    i -> 3
    j -> 4
    k -> 5
    m -> 6
    r -> 7
    q -> 8
    w -> 9
    x -> 10
    e -> 11
    aa -> 12
    ff -> 13
A7: (v, h, bb, j, n, z, dd, ff, o)
    v -> 0
    h -> 1
    bb -> 2
    j -> 3
    n -> 4
    z -> 5
    dd -> 6
    ff -> 7
    o -> 8
A9: (t, f)
    f -> 0
    t -> 1
A10: (t, f)
    f -> 0
    t -> 1
A12: (t, f)
    f -> 0
    t -> 1
A13: (g, p, s)
    g -> 0
    p -> 1
    s -> 2
"""
data["A1"] = data["A1"].replace({"a": 0, "b": 1})
data["A4"] = data["A4"].replace({"u": 0, "y": 1, "l": 2, "t": 3})
data["A5"] = data["A5"].replace({"g": 0, "p": 1, "gg": 2})
data["A6"] = data["A6"].replace({"c": 0, "d": 1, "cc": 2, "i": 3, "j": 4, 
                                 "k": 5, "m": 6, "r": 7, "q": 8, "w": 9, 
                                 "x": 10, "e": 11, "aa": 12, "ff": 13})
data["A7"] = data["A7"].replace({"v": 0, "h": 1, "bb": 2, "j": 3, "n": 4, 
                                 "z": 5, "dd": 6, "ff": 7, "o": 8})
data["A9"] = data["A9"].replace({"f": 0, "t": 1})
data["A10"] = data["A10"].replace({"f": 0, "t": 1})
data["A12"] = data["A12"].replace({"f": 0, "t": 1})
data["A13"] = data["A13"].replace({"g": 0, "p": 1, "s": 2})
print(data)
print(data.dtypes)

#Implementacion del MLPClassifier
X = np.array(data[["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11",
                   "A12","A13","A14","A15"]])#Data
X_TR = X_Entrenamiento(X,n)#Data de Entrenamiento
X_TS = X_Evaluacion(X,n)#Data de Evaluacion

y = np.array(data["A16"])#Objetivos

y_TR = y_Entrenamiento(y, n)#Objetivos de Entrenamiento
y_TS = y_Evaluacion(y, n)#Objetivos de Evaluacion
print("Data de entrenamiento")
print(X_TR)
print("Objetivo de entrenamiento")
print(y_TR)
print("Data de testeo")
print(X_TR)
print("Objetivo de testeo")
print(y_TS)
clasificador = MLPClassifier(tol=1e-2)
clasificador.fit(X_TR, y_TR)
y_PR = clasificador.predict(X_TS)#Objetivos Predecidos con la data de Evaluacion
print("PREDICCION")
print(y_PR)
cm = confusion_matrix(y_TS, y_PR)#Comparacion entre los Objetivos de Testeo y los Objetivos Predecidos
print("MATRIZ DE CONFUSION")
print(cm)