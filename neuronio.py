from random import uniform
import math

class Neuronio:
 
    def __init__(self, numero_atributos, linha, coluna, pesos_aleatorios = False):
        self.w = []
        self.entradas=[]
        self.linha = linha
        self.coluna = coluna
        for i in range (numero_atributos):
            peso = 0
            if pesos_aleatorios is True:
                peso = uniform(0, 1.0)
            self.w.append (peso)
