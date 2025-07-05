import numpy as np
import math
from sklearn.datasets import load_iris # Para carregar o dataset Iris

# Classe Neuronio: Representa um único neurônio no mapa SOM
class Neuronio:
    def __init__(self, numero_atributos, linha, coluna, pesos_aleatorios=True):
        """
        Inicializa um neurônio no mapa SOM.

        Args:
            numero_atributos (int): Número de atributos do vetor de pesos (mesma dimensão da entrada).
            linha (int): Posição da linha do neurônio no grid.
            coluna (int): Posição da coluna do neurônio no grid.
            pesos_aleatorios (bool): Se True, inicializa os pesos com valores aleatórios entre 0 e 1.
                                     Caso contrário, inicializa com zeros.
        """
        self.w = [] # Vetor de pesos do neurônio
        self.linha = linha
        self.coluna = coluna
        for _ in range(numero_atributos):
            peso = 0
            if pesos_aleatorios:
                peso = np.random.uniform(0, 1.0) # Usa numpy para uniform
            self.w.append(peso)
        self.w = np.array(self.w) # Converte a lista de pesos para array numpy para facilitar operações

# Classe SOM: Implementa o algoritmo Self-Organizing Map
class SOM:
    def __init__(self, linhas, colunas, iteracoes, taxa_aprendizado_inicial, sigma_inicial, decaimento='linear'):
        """
        Inicializa o mapa SOM.

        Args:
            linhas (int): Número de linhas do grid SOM.
            colunas (int): Número de colunas do grid SOM.
            iteracoes (int): Número total de iterações (épocas) de treinamento (T).
            taxa_aprendizado_inicial (float): Taxa de aprendizado inicial (eta_0).
            sigma_inicial (float): Raio de vizinhança inicial (sigma_0).
            decaimento (str): Estratégia de decaimento para eta e sigma ('constante', 'linear', 'nao-linear').
        """
        self.linhas = linhas
        self.colunas = colunas
        self.iteracoes = iteracoes
        self.taxa_aprendizado_inicial = taxa_aprendizado_inicial
        self.sigma_inicial = sigma_inicial
        self.decaimento = decaimento
        self.matriz = [] # O grid de neurônios
        self.entradas_treinamento = None # Será preenchido com os dados de entrada normalizados

    def _inicializar_pesos(self, numero_atributos):
        """Inicializa os pesos de todos os neurônios no grid."""
        for i in range(self.linhas):
            self.matriz.append([])
            for j in range(self.colunas):
                # Passa True para pesos_aleatorios para inicialização aleatória
                self.matriz[i].append(Neuronio(numero_atributos, linha=i, coluna=j, pesos_aleatorios=True))

    def distancia_euclidiana(self, vetor1, vetor2):
        """
        Calcula a distância Euclidiana entre dois vetores.
        Corresponde a d_j = sqrt(sum((x_k - w_j,k)^2)) na formulação matemática.
        """
        return np.linalg.norm(vetor1 - vetor2)

    def calcular_bmu(self, entrada):
        """
        Encontra a Unidade de Melhor Correspondência (BMU) para uma dada entrada.
        Corresponde a BMU = argmin_j(d_j) na formulação matemática.
        """
        bmu = None
        menor_distancia = float('inf')

        for linha in self.matriz:
            for neuronio in linha:
                distancia_atual = self.distancia_euclidiana(entrada, neuronio.w)
                if distancia_atual < menor_distancia:
                    menor_distancia = distancia_atual
                    bmu = neuronio
        return bmu

    def _funcao_vizinhanca_gaussiana(self, r, sigma):
        """
        Calcula a função de vizinhança gaussiana (theta).
        Corresponde a theta(j*, j) = e^(-r / (2 * sigma^2)) na formulação matemática.
        """
        return math.exp(-r / (2 * sigma**2))

    def _atualizar_pesos(self, bmu, entrada, eta, sigma):
        """
        Atualiza os pesos de todos os neurônios no mapa.
        Corresponde a w_j_novo = w_j_antigo + theta(j*, j) * (x - w_j_antigo) * eta na formulação.
        """
        for linha_mapa in self.matriz:
            for neuronio_atual in linha_mapa:
                # Distância topológica (r) entre o BMU e o neurônio atual no grid
                # r = sqrt((j*_linha - j_linha)^2 + (j*_coluna - j_coluna)^2)
                r_topologica = self.distancia_euclidiana(
                    np.array([bmu.linha, bmu.coluna]),
                    np.array([neuronio_atual.linha, neuronio_atual.coluna])
                )

                # Calcula a taxa de vizinhança (theta)
                taxa_vizinhanca = self._funcao_vizinhanca_gaussiana(r_topologica, sigma)

                # Aplica a regra de atualização dos pesos
                neuronio_atual.w += taxa_vizinhanca * (entrada - neuronio_atual.w) * eta

    def _calcular_eta_k(self, k):
        """Calcula a taxa de aprendizado (eta) para a iteração k."""
        if self.decaimento == 'constante':
            return self.taxa_aprendizado_inicial
        elif self.decaimento == 'linear':
            return self.taxa_aprendizado_inicial * (1 - (k / self.iteracoes)) [cite: 1]
        elif self.decaimento == 'nao-linear':
            # Tau (constante de tempo) pode ser ajustado, aqui usando T/2 como exemplo 
            tau = self.iteracoes / 2
            return self.taxa_aprendizado_inicial * math.exp(-k / tau) [cite: 1]
        else:
            raise ValueError("Estratégia de decaimento inválida para eta.")

    def _calcular_sigma_k(self, k):
        """Calcula o raio de vizinhança (sigma) para a iteração k."""
        if self.decaimento == 'constante':
            return self.sigma_inicial
        elif self.decaimento == 'linear':
            # Sigma também pode decair linearmente
            return self.sigma_inicial * (1 - (k / self.iteracoes))
        elif self.decaimento == 'nao-linear':
            # Tau (constante de tempo) para sigma também pode ser ajustado
            tau = self.iteracoes / 2
            return self.sigma_inicial * math.exp(-k / tau) [cite: 1]
        else:
            raise ValueError("Estratégia de decaimento inválida para sigma.")

    def treinar(self, dados_entrada):
        """
        Realiza o treinamento do SOM.

        Args:
            dados_entrada (np.ndarray): O conjunto de dados de entrada a ser treinado.
                                        Deve estar normalizado.
        """
        self.entradas_treinamento = dados_entrada
        numero_atributos = dados_entrada.shape[1] # Pega a dimensão dos vetores de entrada
        self._inicializar_pesos(numero_atributos) # Inicializa os pesos dos neurônios

        print(f"Iniciando treinamento do SOM com {self.iteracoes} iterações...")
        for k in range(self.iteracoes):
            # Calcular eta_k e sigma_k para a iteração atual
            eta_k = self._calcular_eta_k(k)
            sigma_k = self._calcular_sigma_k(k)

            # Para cada entrada no conjunto de treinamento
            for entrada in self.entradas_treinamento:
                bmu = self.calcular_bmu(entrada) # Encontra o BMU para a entrada
                self._atualizar_pesos(bmu, entrada, eta_k, sigma_k) # Atualiza os pesos da rede

            if (k + 1) % (self.iteracoes // 10) == 0 or k == self.iteracoes -1:
                print(f"Iteração {k+1}/{self.iteracoes} - eta: {eta_k:.4f}, sigma: {sigma_k:.4f}")
        print("Treinamento concluído.")

# --- Funções Auxiliares para Carregamento e Normalização de Dados ---
def carregar_e_normalizar_iris():
    """
    Carrega o dataset Iris e normaliza seus atributos entre 0 e 1.
    """
    iris = load_iris()
    dados = iris.data # Atributos (sepal length, sepal width, petal length, petal width)

    # Normalização Min-Max
    min_vals = dados.min(axis=0)
    max_vals = dados.max(axis=0)
    # Evitar divisão por zero se max_vals == min_vals para alguma coluna
    dados_normalizados = (dados - min_vals) / (max_vals - min_vals + 1e-10)
    
    return dados_normalizados, iris.target, iris.feature_names, iris.target_names



    
