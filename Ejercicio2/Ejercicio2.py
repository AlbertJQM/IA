# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:06:07 2020

@author: Albert
"""
import random

modeloFinal = [1,1,1,1,1,1,1,1,1,1] 
tamañoIndividuo = 10

num = 200 #Cantidad de individuos
gen = 100 #Generaciones
pressure = 3 #individual>2
probabilidadMutacion = 0.2

def individuo(min, max):
    return[random.randint(min, max) for i in range(tamañoIndividuo)]

def newPoblacion():
    return [individuo(0,1) for i in range(num)]

# Funcion la que se debe cambiar en funcion a decimal f(x)
def funcion(individuo):
    decimal = int("".join(map(str, individuo)),2)
    y = decimal**3 + decimal ** 2 + decimal
    return y,

def seleccionReproduction(poblacion):
    evaluacion = [ (funcion(i), i) for i in poblacion]
    print("eval",evaluacion)
    evaluacion = [i[1] for i in sorted(evaluacion)]
    print("eval",evaluacion)
    poblacion = evaluacion
    selected = evaluacion[(len(evaluacion)-pressure):]
    for i in range(len(poblacion)-pressure):
        
        puntoCambio = random.randint(1,tamañoIndividuo-1)
        padre = random.sample(selected, 2)
        poblacion[i][:puntoCambio] = padre[0][:puntoCambio]
        poblacion[i][puntoCambio:] = padre[1][puntoCambio:]
        
        print("-------------")
        print(padre[0])
        print(padre[1])
        print(puntoCambio)
        print(poblacion[i])
        
    return poblacion

def mutacion(poblacion):
    for i in range(len(poblacion)-pressure):
        if random.random() <= probabilidadMutacion: 
            puntoCambio = random.randint(1,tamañoIndividuo-1) 
            new_val = random.randint(0,9) 
            print("--")
            print(poblacion[i])
            while new_val == poblacion[i][puntoCambio]:
                new_val = random.randint(0,9)
            poblacion[i][puntoCambio] = new_val
            print(puntoCambio)
            print(poblacion[i])
    return poblacion


# Principal
poblacion = newPoblacion()
print("\nPoblacion Inicial:\n%s"%(poblacion))
poblacion = seleccionReproduction(poblacion)
print("\nSeleccion de Poblacion:\n%s"%(poblacion))
poblacion = mutacion(poblacion)
print("\nMutacion de la Poblacion:\n%s"%(poblacion))
