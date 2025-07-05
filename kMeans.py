import numpy as np
import math
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt # Importa matplotlib para visualização
import pandas as pd
class KMeans:
    """
    Implementação do algoritmo K-means para agrupamento (clusterização) de dados.
    """
    def __init__(self, numero_clusters=3, max_iteracoes=200, tolerancia=1e-5, plotar_passos=False):
        """
        Inicializa o algoritmo K-means com os parâmetros definidos.

        Args:
            numero_clusters (int): O número de clusters (k) desejado para agrupar os dados.
            max_iteracoes (int): O número máximo de iterações permitidas para o algoritmo.
            tolerancia (float): Um valor pequeno que define o critério de parada.
                                Se a distância que os centroides se movem entre iterações for menor que este valor,
                                o algoritmo é considerado convergido.
            plotar_passos (bool): Se verdadeiro, tenta plotar o estado dos agrupamentos a cada iteração.
        """
        self.numero_clusters = numero_clusters
        self.max_iteracoes = max_iteracoes
        self.tolerancia = tolerancia
        self.plotar_passos = plotar_passos
        self.centroides = None # Armazenará os centroides dos clusters
        self.agrupamentos = None # Armazenará os índices dos pontos de dados pertencentes a cada cluster
        self.dados = None # Armazenará os dados de entrada para o treinamento

    def distancia_euclidiana(self, vetor1, vetor2):
        """
        Calcula a distância Euclidiana entre dois vetores (pontos no espaço).
        """
        return np.linalg.norm(vetor1 - vetor2)

    def _atribuir_ponto_ao_cluster(self, ponto_dado, centroides_atuais):
        """
        Atribui um ponto de dado ao centroide mais próximo, calculando a distância
        deste ponto a todos os centroides atuais.
        """
        distancias = [self.distancia_euclidiana(ponto_dado, c) for c in centroides_atuais]
        indice_mais_proximo = np.argmin(distancias)
        return indice_mais_proximo

    def _criar_agrupamentos(self, dados_entrada, centroides_atuais):
        """
        Organiza os pontos de dados em clusters, atribuindo cada ponto ao seu centroide mais próximo.
        Returna uma lista de listas, onde cada sublista contém os índices dos pontos que foram atribuídos a um cluster específico.
        """
        agrupamentos = [[] for _ in range(self.numero_clusters)]
        for indice_ponto, ponto_dado in enumerate(dados_entrada):
            indice_centroide = self._atribuir_ponto_ao_cluster(ponto_dado, centroides_atuais)
            agrupamentos[indice_centroide].append(indice_ponto)
        return agrupamentos

    def _atualizar_centroides(self, dados_entrada, agrupamentos_atuais):
        """
        Recalcula as posições dos centroides, utilizando a média dos pontos de dados
        atribuídos a cada cluster.
        """
        novos_centroides = np.zeros((self.numero_clusters, dados_entrada.shape[1]))
        for idx, indices_pontos in enumerate(agrupamentos_atuais):
            if len(indices_pontos) > 0: # Garante que o cluster não esteja vazio para evitar erro de divisão por zero
                pontos_do_cluster = dados_entrada[indices_pontos]
                novos_centroides[idx] = np.mean(pontos_do_cluster, axis=0)
            else:
                # Se um cluster ficar vazio, o centroide correspondente não é movido
                novos_centroides[idx] = self.centroides[idx]
        return novos_centroides

    def treinar(self, dados_treinamento):
        """
        Executa o algoritmo K-means para agrupar os dados de treinamento.

        """
        self.dados = dados_treinamento # Armazena os dados para uso interno na instância

        # 1. Inicialização: Seleciona k centroides iniciais aleatoriamente a partir dos próprios dados
        indices_iniciais = np.random.choice(self.dados.shape[0], self.numero_clusters, replace=False)
        self.centroides = self.dados[indices_iniciais]

        # Loop principal de iterações para refinamento dos clusters
        for iteracao in range(self.max_iteracoes):
            # Armazena os centroides da iteração anterior para verificar a convergência
            centroides_anteriores = self.centroides.copy()

            # 2. Fase de Atribuição: Atribui cada ponto de dado ao cluster cujo centroide é o mais próximo
            self.agrupamentos = self._criar_agrupamentos(self.dados, self.centroides)

            # 3. Fase de Atualização: Recalcula a posição dos centroides com base nos pontos atribuídos
            self.centroides = self._atualizar_centroides(self.dados, self.agrupamentos)

            # 4. Critério de Parada: Verifica a convergência pela distância de movimento dos centroides
            movimento_maximo_centroide = 0.0
            for i in range(self.numero_clusters):
                distancia_movida = self.distancia_euclidiana(self.centroides[i], centroides_anteriores[i])
                if distancia_movida > movimento_maximo_centroide:
                    movimento_maximo_centroide = distancia_movida

            # Se o maior movimento de um centroide for menor que a tolerância, o algoritmo convergiu
            if movimento_maximo_centroide < self.tolerancia:
                print(f"K-means convergiu na iteração {iteracao + 1}. Movimento máximo do centroide: {movimento_maximo_centroide:.6f}")
                break

            # Opcional: plotar o estado dos agrupamentos a cada passo (se a flag plotar_passos for True e for 2D)
            if self.plotar_passos and self.dados.shape[1] == 2:
                self._plotar_agrupamentos(f"Iteração {iteracao + 1}")

        # Se o loop terminar por atingir max_iteracoes sem convergir
        if iteracao == self.max_iteracoes - 1 and movimento_maximo_centroide >= self.tolerancia:
            print(f"K-means atingiu o número máximo de iterações ({self.max_iteracoes}) sem convergir totalmente.")

        return self.centroides, self.agrupamentos

def carregar_e_normalizar_iris():
    """
    Carrega o dataset Iris da biblioteca scikit-learn e normaliza seus atributos
    para o intervalo entre 0 e 1 (Normalização Min-Max).

    Returns:
        np.array: O dataset Iris normalizado, pronto para ser utilizado pelo K-means.
    """
    iris = load_iris()
    dados = iris.data # Carrega os atributos do dataset Iris

    # Realiza a Normalização Min-Max para cada atributo
    min_vals = dados.min(axis=0) # Valores mínimos de cada coluna
    max_vals = dados.max(axis=0) # Valores máximos de cada coluna

    # Evita divisão por zero: adiciona um pequeno valor (epsilon) no denominador
    # caso max_vals seja igual a min_vals para alguma coluna (atributo constante)
    dados_normalizados = (dados - min_vals) / (max_vals - min_vals + 1e-10)

    return dados_normalizados


from sklearn.metrics import adjusted_rand_score
def visualizar_coordenadas_paralelas(dados, classes, nomes_atributos=None):
    """
    Gera um gráfico de coordenadas paralelas para visualizar dados de alta dimensão,
    com as linhas coloridas de acordo com as classes dos pontos.
    Usa um colormap personalizado com cores fixas para garantir consistência.
    """
    if nomes_atributos is None:
        nomes_atributos = [f'Atributo_{i}' for i in range(dados.shape[1])]

    df = pd.DataFrame(dados, columns=nomes_atributos)
    df['Classe'] = classes

    plt.figure(figsize=(10, 7))


    cores_fixas = ['#4B0082', '#008080', '#FFFF00']
    custom_colormap = ListedColormap(cores_fixas)
    pd.plotting.parallel_coordinates(df, 'Classe', colormap=custom_colormap)

    plt.title('Visualização de Coordenadas Paralelas')
    plt.xlabel('Atributos')
    plt.ylabel('Valores Normalizados (para visualização)')
    plt.grid(True)
    plt.show()

def calcular_ari(rotulos_verdadeiros, rotulos_preditos):
    """
    Calcula o Adjusted Rand Index (ARI).
    O ARI é uma medida de similaridade entre duas atribuições de clusterings, ajustada para o acaso.
    Um valor de 1.0 indica agrupamentos idênticos, 0.0 indica agrupamentos aleatórios.
    """
    # Usamos a função do scikit-learn, que é robusta.
    return adjusted_rand_score(rotulos_verdadeiros, rotulos_preditos)

# --- Execução do K-means ---
if __name__ == "__main__":
    # Carrega o dataset Iris e o normaliza
    iris_dataset = load_iris()
    dados_iris_original = iris_dataset.data
    classes_iris_original = iris_dataset.target # As classes originais do Iris (0, 1, 2)
    nomes_atributos_iris = iris_dataset.feature_names

    dados_iris_normalizados = carregar_e_normalizar_iris()

    print("--- Testando o K-means ---")
    kmeans_instancia = KMeans(numero_clusters=3, max_iteracoes=300, tolerancia=1e-4, plotar_passos=False)
    centroides_finais, agrupamentos_finais = kmeans_instancia.treinar(dados_iris_normalizados)

    print("\n--- Resultados Finais do K-means ---")
    print("Centroides Finais (normalizados):\n", centroides_finais)

    # --- Preparação dos rótulos do K-means para visualização ---
    rotulos_kmeans = np.zeros(dados_iris_normalizados.shape[0], dtype=int)
    for cluster_id, indices_pontos in enumerate(agrupamentos_finais):
        for idx in indices_pontos:
            rotulos_kmeans[idx] = cluster_id

    ari = calcular_ari(classes_iris_original, rotulos_kmeans)
    print(f"Adjusted Rand Index (ARI) do Agrupamento K-means: {ari:.4f}")
   

    print("\n--- Visualização dos Agrupamentos do K-means (Coordenadas Paralelas, Cores Ajustadas) ---")
    # Passamos os rótulos do K-means JÁ REMAPEADOS para que as cores fiquem consistentes.
    visualizar_coordenadas_paralelas(dados_iris_normalizados, rotulos_kmeans, nomes_atributos_iris)


    print("\n--- Visualização das Classes Originais do Iris (Coordenadas Paralelas) ---")
    # Visualizamos as classes originais usando os mesmos dados normalizados e nomes de atributos.
    visualizar_coordenadas_paralelas(dados_iris_normalizados, classes_iris_original, nomes_atributos_iris)
