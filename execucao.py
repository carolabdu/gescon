
# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Carregar e normalizar os dados Iris
    dados_iris_normalizados, labels_iris, feature_names_iris, target_names_iris = carregar_e_normalizar_iris()
    print("Shape dos dados normalizados:", dados_iris_normalizados.shape)

    # --- Definição dos 10 Cenários de Hiperparâmetros ---
    scenarios = [
        {"name": "Cenário 1 (Base, Linear)", "linhas": 4, "colunas": 4, "iteracoes": 100, "taxa_aprendizado_inicial": 0.5, "sigma_inicial": 2.0, "decaimento": 'linear'},
        {"name": "Cenário 2 (Médio, Não-Linear)", "linhas": 7, "colunas": 7, "iteracoes": 200, "taxa_aprendizado_inicial": 0.3, "sigma_inicial": 3.5, "decaimento": 'nao-linear'},
        {"name": "Cenário 3 (Grande, Linear)", "linhas": 10, "colunas": 10, "iteracoes": 300, "taxa_aprendizado_inicial": 0.4, "sigma_inicial": 5.0, "decaimento": 'linear'},
        {"name": "Cenário 4 (Retangular, Não-Linear, LR Alto)", "linhas": 6, "colunas": 10, "iteracoes": 150, "taxa_aprendizado_inicial": 0.6, "sigma_inicial": 5.0, "decaimento": 'nao-linear'},
        {"name": "Cenário 5 (Pequeno, Rápido, LR Alto)", "linhas": 3, "colunas": 3, "iteracoes": 50, "taxa_aprendizado_inicial": 0.7, "sigma_inicial": 1.5, "decaimento": 'linear'},
        {"name": "Cenário 6 (Médio-Grande, Lento, LR Baixo)", "linhas": 8, "colunas": 8, "iteracoes": 500, "taxa_aprendizado_inicial": 0.2, "sigma_inicial": 4.0, "decaimento": 'nao-linear'},
        {"name": "Cenário 7 (Constante)", "linhas": 5, "colunas": 5, "iteracoes": 100, "taxa_aprendizado_inicial": 0.5, "sigma_inicial": 2.5, "decaimento": 'constante'},
        {"name": "Cenário 8 (Grande, Sigma Menor)", "linhas": 9, "colunas": 9, "iteracoes": 250, "taxa_aprendizado_inicial": 0.4, "sigma_inicial": 3.0, "decaimento": 'linear'},
        {"name": "Cenário 9 (Retangular, Não-Linear, Sigma Maior)", "linhas": 5, "colunas": 8, "iteracoes": 200, "taxa_aprendizado_inicial": 0.3, "sigma_inicial": 5.0, "decaimento": 'nao-linear'},
        {"name": "Cenário 10 (Muito Grande, Rápido, LR Alto)", "linhas": 12, "colunas": 12, "iteracoes": 100, "taxa_aprendizado_inicial": 0.7, "sigma_inicial": 6.0, "decaimento": 'linear'},
    ]

    # --- Execução de cada cenário ---
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Executando {scenario['name']} (Cenário {i+1}/10) ---")
        
        som_model = SOM(
            linhas=scenario['linhas'],
            colunas=scenario['colunas'],
            iteracoes=scenario['iteracoes'],
            taxa_aprendizado_inicial=scenario['taxa_aprendizado_inicial'],
            sigma_inicial=scenario['sigma_inicial'],
            decaimento=scenario['decaimento']
        )
        
        # Treinar o modelo (usar uma cópia dos dados para garantir que não haja side-effects entre as rodadas)
        som_model.treinar(dados_iris_normalizados.copy())
        
        # Calcular e exibir o erro de quantização
        erro_quantizacao = som_model.calcular_erro_quantizacao()
        print(f"Erro de Quantização para {scenario['name']}: {erro_quantizacao:.4f}")
        
        # Gerar e exibir as visualizações
        som_model.visualizar_umatrix(title=f"U-Matrix - {scenario['name']}")
        som_model.visualizar_hit_map(labels=labels_iris, title=f"Hit Map - {scenario['name']}")
        som_model.visualizar_component_planes(feature_names=feature_names_iris, title=f"Planos de Componentes - {scenario['name']}")

    print("\nExecução de todos os 10 cenários concluída.")
    print("Analise os Erros de Quantização e as visualizações para comparar o impacto dos diferentes hiperparâmetros.")
