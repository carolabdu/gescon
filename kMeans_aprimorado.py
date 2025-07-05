import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn.metrics import adjusted_rand_score
import matplotlib.lines as mlines # Necessário para a função de visualização de coordenadas paralelas, mesmo que a cor seja ignorada


# --- Definição da Classe KMeans ---
class KMeans:
    """
    Implementação do algoritmo K-means para agrupamento (clusterização) de dados.
    """
    def __init__(self, numero_clusters=3, max_iteracoes=100, tolerancia=1e-4, plotar_passos=False):
        self.numero_clusters = numero_clusters
        self.max_iteracoes = max_iteracoes
        self.tolerancia = tolerancia
        self.plotar_passos = plotar_passos
        self.centroides = None
        self.agrupamentos = None
        self.dados = None
        self.historico_sse = []

    def distancia_euclidiana(self, vetor1, vetor2):
        return np.linalg.norm(vetor1 - vetor2)

    def _atribuir_ponto_ao_cluster(self, ponto_dado, centroides_atuais):
        distancias = [self.distancia_euclidiana(ponto_dado, c) for c in centroides_atuais]
        indice_mais_proximo = np.argmin(distancias)
        return indice_mais_proximo

    def _criar_agrupamentos(self, dados_entrada, centroides_atuais):
        agrupamentos = [[] for _ in range(self.numero_clusters)]
        for indice_ponto, ponto_dado in enumerate(dados_entrada):
            indice_centroide = self._atribuir_ponto_ao_cluster(ponto_dado, centroides_atuais)
            agrupamentos[indice_centroide].append(indice_ponto)
        return agrupamentos

    def _atualizar_centroides(self, dados_entrada, agrupamentos_atuais):
        novos_centroides = np.zeros((self.numero_clusters, dados_entrada.shape[1]))
        for idx, indices_pontos in enumerate(agrupamentos_atuais):
            if len(indices_pontos) > 0:
                pontos_do_cluster = dados_entrada[indices_pontos]
                novos_centroides[idx] = np.mean(pontos_do_cluster, axis=0)
            else:
                novos_centroides[idx] = self.centroides[idx] # Mantém o centroide se o cluster ficar vazio
        return novos_centroides

    def treinar(self, dados_treinamento):
        self.dados = dados_treinamento
        self.historico_sse = []

        indices_iniciais = np.random.choice(self.dados.shape[0], self.numero_clusters, replace=False)
        self.centroides = self.dados[indices_iniciais]

        for iteracao in range(self.max_iteracoes):
            centroides_anteriores = self.centroides.copy()
            self.agrupamentos = self._criar_agrupamentos(self.dados, self.centroides)
            self.centroides = self._atualizar_centroides(self.dados, self.agrupamentos)

            current_sse = 0
            for cluster_id, indices_pontos in enumerate(self.agrupamentos):
                if indices_pontos:
                    pontos_do_cluster = self.dados[indices_pontos]
                    centroid = self.centroides[cluster_id]
                    current_sse += np.sum((pontos_do_cluster - centroid)**2)
            self.historico_sse.append(current_sse)

            movimento_maximo_centroide = 0.0
            for i in range(self.numero_clusters):
                distancia_movida = self.distancia_euclidiana(self.centroides[i], centroides_anteriores[i])
                if distancia_movida > movimento_maximo_centroide:
                    movimento_maximo_centroide = distancia_movida

            if movimento_maximo_centroide < self.tolerancia:
                break

            if self.plotar_passos and self.dados.shape[1] == 2:
                self._plotar_agrupamentos(f"Iteração {iteracao + 1}")

        if iteracao == self.max_iteracoes - 1 and movimento_maximo_centroide >= self.tolerancia:
            pass
        
        return self.centroides, self.agrupamentos, self.historico_sse[-1] if self.historico_sse else float('inf')

    def _plotar_agrupamentos(self, titulo="Agrupamentos K-means"):
        if self.dados.shape[1] != 2:
            print("Visualização de passos é possível apenas para dados 2D.")
            return

        plt.figure(figsize=(8, 6))
        cores = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown']

        for idx, indices_pontos in enumerate(self.agrupamentos):
            if indices_pontos:
                pontos = self.dados[indices_pontos]
                plt.scatter(pontos[:, 0], pontos[:, 1], color=cores[idx % len(cores)], label=f'Cluster {idx + 1}', alpha=0.6)
        
        plt.scatter(self.centroides[:, 0], self.centroides[:, 1], marker='X', s=200, color='black', edgecolor='white', linewidth=1.5, label='Centroides')
        
        plt.title(titulo)
        plt.xlabel('Atributo 1')
        plt.ylabel('Atributo 2')
        plt.legend()
        plt.grid(True)
        plt.show()

# --- Funções Auxiliares ---
def carregar_e_normalizar_iris():
    iris = load_iris()
    dados = iris.data

    min_vals = dados.min(axis=0)
    max_vals = dados.max(axis=0)
    
    dados_normalizados = (dados - min_vals) / (max_vals - min_vals + 1e-10)

    return dados_normalizados

# Função de Visualização de Coordenadas Paralelas (Versão anterior, com cores fixas mas sem o tratamento extra da legenda)
def visualizar_coordenadas_paralelas(dados, classes, nomes_atributos=None):
    if nomes_atributos is None:
        nomes_atributos = [f'Atributo_{i}' for i in range(dados.shape[1])]
    
    df = pd.DataFrame(dados, columns=nomes_atributos)
    df['Classe'] = classes # A coluna 'Classe' pode conter os IDs brutos ou mapeados

    plt.figure(figsize=(10, 7))

    # --- Cores Personalizadas para as Classes (0, 1, 2) ---
    # Assegura que o colormap tenha cores fixas para os valores 0, 1, 2.
    # Esta parte é mantida pois o K-means mapeia para essas 3 classes.
    cores_fixas = ['#4B0082', '#008080', '#FFFF00'] # Roxo/Índigo, Verde-água/Teal, Amarelo
    custom_colormap = ListedColormap(cores_fixas)

    # Usa o colormap personalizado na função parallel_coordinates
    # A legenda será gerada automaticamente pelo Pandas, que pode ter a ordem "invertida"
    pd.plotting.parallel_coordinates(df, 'Classe', colormap=custom_colormap) 

    plt.title('Visualização de Coordenadas Paralelas')
    plt.xlabel('Atributos')
    plt.ylabel('Valores Normalizados (para visualização)')
    plt.grid(True)
    plt.show()

# --- Funções de Plotagem e Métricas ---
def plotar_convergencia_sse(sse_history, titulo="Convergência SSE do K-means"):
    if not sse_history:
        print("Histórico de SSE não disponível.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sse_history) + 1), sse_history, marker='o', linestyle='-')
    plt.title(titulo)
    plt.xlabel('Número da Iteração')
    plt.ylabel('Soma dos Quadrados dos Erros (SSE/Inércia)')
    plt.grid(True)
    plt.show()

def calcular_pureza(rotulos_verdadeiros, rotulos_preditos):
    if len(rotulos_verdadeiros) != len(rotulos_preditos):
        raise ValueError("Os arrays de rótulos devem ter o mesmo comprimento.")

    total_pontos = len(rotulos_verdadeiros)
    if total_pontos == 0:
        return 0.0

    cluster_class_counts = {}
    for true_label, predicted_label in zip(rotulos_verdadeiros, rotulos_preditos):
        if predicted_label not in cluster_class_counts:
            cluster_class_counts[predicted_label] = {}
        if true_label not in cluster_class_counts[predicted_label]:
            cluster_class_counts[predicted_label][true_label] = 0
        cluster_class_counts[predicted_label][true_label] += 1

    soma_maior_contagem = 0
    for cluster_label in cluster_class_counts:
        if cluster_class_counts[cluster_label]:
            soma_maior_contagem += max(cluster_class_counts[cluster_label].values())
    
    pureza = soma_maior_contagem / total_pontos
    return pureza

def calcular_ari(rotulos_verdadeiros, rotulos_preditos):
    return adjusted_rand_score(rotulos_verdadeiros, rotulos_preditos)


# --- Bloco Principal para Teste (com múltiplas rodadas) ---
if __name__ == "__main__":
    # Carrega o dataset Iris e o normaliza
    iris_dataset = load_iris()
    dados_iris_original = iris_dataset.data
    classes_iris_original = iris_dataset.target # As classes originais do Iris (0, 1, 2)
    nomes_atributos_iris = iris_dataset.feature_names

    dados_iris_normalizados = carregar_e_normalizar_iris()

    print("--- Testando o K-means com Múltiplas Inicializações ---")
    
    num_runs = 20 # Número de vezes para rodar o K-means
    best_sse = float('inf')
    best_centroides = None
    best_agrupamentos = None
    best_rotulos_kmeans = None
    best_historico_sse = None # Para armazenar o histórico SSE da melhor execução

    for i in range(num_runs):
        # print(f"\n--- Execução {i+1}/{num_runs} ---") # Opcional: para ver o progresso
        kmeans_instancia = KMeans(numero_clusters=3, max_iteracoes=300, tolerancia=1e-4, plotar_passos=False)
        
        # Treina e captura o SSE final desta execução
        current_centroides, current_agrupamentos, current_final_sse = kmeans_instancia.treinar(dados_iris_normalizados)

        # Se esta execução obteve um SSE melhor, armazena seus resultados
        if current_final_sse < best_sse:
            best_sse = current_final_sse
            best_centroides = current_centroides
            best_agrupamentos = current_agrupamentos
            best_historico_sse = kmeans_instancia.historico_sse # Salva o histórico completo da melhor run

            # Reconstrói os rótulos do K-means para esta melhor execução
            temp_rotulos_kmeans = np.zeros(dados_iris_normalizados.shape[0], dtype=int)
            for cluster_id, indices_pontos in enumerate(best_agrupamentos):
                for idx in indices_pontos:
                    temp_rotulos_kmeans[idx] = cluster_id
            best_rotulos_kmeans = temp_rotulos_kmeans

    print("\n--- Resultados da Melhor Execução do K-means (Após Múltiplas Tentativas) ---")
    print(f"Melhor SSE (Inércia) encontrada: {best_sse:.4f}")
    print("Centroides Finais da Melhor Execução (normalizados):\n", best_centroides)

    # --- Plotar a Convergência do SSE da MELHOR execução ---
    print("\n--- Plotando Convergência SSE da Melhor Execução do K-means ---")
    plotar_convergencia_sse(best_historico_sse) # Passa o histórico da melhor run

    # --- Avaliar a Taxa de Acerto (Pureza e ARI) da MELHOR execução ---
    print("\n--- Avaliação da Taxa de Acerto (Comparando Melhor K-means com Classes Originais) ---")
    # Utilizamos os rótulos do K-means da melhor execução para as métricas
    pureza = calcular_pureza(classes_iris_original, best_rotulos_kmeans)
    ari = calcular_ari(classes_iris_original, best_rotulos_kmeans)

    print(f"Pureza do Agrupamento K-means (Melhor Execução): {pureza:.4f}")
    print(f"Adjusted Rand Index (ARI) do Agrupamento K-means (Melhor Execução): {ari:.4f}")
  

    print("\n--- Visualização dos Agrupamentos do K-means (Melhor Execução, Coordenadas Paralelas, Cores Ajustadas) ---")
    visualizar_coordenadas_paralelas(dados_iris_normalizados, best_rotulos_kmeans, nomes_atributos_iris)


    print("\n--- Visualização das Classes Originais do Iris (Coordenadas Paralelas) ---")
    visualizar_coordenadas_paralelas(dados_iris_normalizados, classes_iris_original, nomes_atributos_iris)
