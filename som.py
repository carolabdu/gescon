from neuronio.py import neuronio

def __init__(self, entradas, linhas, colunas, iteracoes, taxa_aprendizado, decaimento='linear'):
        self.iteracoes = iteracoes
        self.entradas = entradas #entradas já normalizadas
        self.taxa_aprendizado = taxa_aprendizado
        self.matriz = []
        self.linhas = linhas
        self.colunas = colunas
        self.decaimento = decaimento
        self.sigma_vizinhanca = 1
        for i in range(linhas):
            self.matriz.append([])
            for j in range (colunas):
                self.matriz [i].append (Neuronio (numero_atributos=len(entradas),linha=i, coluna=j, pesos_aleatorios=True))

  def calculo_bmu(self):
    bmu = None
    d_bmu =0
    for l in range(self.linhas):
      for c in range(self.colunas):
        dj_sum = 0
        neuronio = self.matriz[l][c]
        total_pesos = len(self.entradas)
        for i in range(total_pesos):
          dej_sum += (self.entrada[i] - neuronio.w[i])**2
        dj = math.sqrt(dj_sum)
        if bmu is None or d < d_bmu:
            d_bmu = d
            bmu = neuronio
      bmu.entradas.append(entrada)
      return bmu
