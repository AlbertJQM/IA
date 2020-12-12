# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 23:17:18 2020

@author: Albert
"""
import pandas as pd
import numpy as np
import random
import numpy
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
#Cargando el archivo CSV con las distancias
df = pd.read_csv("distancias.csv")
print("Matriz de distancias")
print(df)

#Convirtiendo el df a matriz
matriz = np.array(df)
av = {
"Cant_Lugares" : len(matriz),#Se obtiene la cantidad de lugares que se visitará
"Matriz_Distancia" : matriz  
}

MatrizDistancia = av["Matriz_Distancia"] 
IndLugares = av["Cant_Lugares"]

print(MatrizDistancia)
print(IndLugares)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

toolbox.register("indices", random.sample, range(IndLugares), IndLugares)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
ind = toolbox.individual() # creamos un individuo aleatorio

pop = toolbox.population(n=300) # creamos una población aleatoria

def evalAV(individual):
    """ Función objetivo, calcula la distancia que recorre el viajante"""
    # distancia entre el último elemento y el primero
    distancia = MatrizDistancia[individual[-1]][individual[0]]
    # distancia entre el resto de ciudades
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distancia += MatrizDistancia[gene1][gene2]
    return distancia,

toolbox.register("evaluate", evalAV)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)    
    
def main():
    random.seed(64) # ajuste de la semilla del generador de números aleatorios
    pop = toolbox.population(n=300) # creamos la población inicial 
    hof = tools.HallOfFame(1) 
    stats = tools.Statistics(lambda ind: ind.fitness.values) 
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    log = tools.Logbook()     
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100
                                       , stats=stats, halloffame=hof, verbose=False)
    return pop, hof, log
if __name__ == "__main__":
    pop, hof, log = main()
    print(log)
    print("Mejor fitness: %f" %hof[0].fitness.values)
    print("Mejor individuo: %s" %hof[0])