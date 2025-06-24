from neuronio.py import neuronio


class SOM:

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

  def calculo_bmu(self,entradas):
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
        if bmu is None or dj < d_bmu:
            d_bmu = dj
            bmu = neuronio
      bmu.entradas.append(entrada)
      return bmu

def ajuste_pesos(self,entrada bmu, eta, sigma):
        for l in self.matriz: 
                for neuronio in l:
                        r = math.sqrt(((bmu.linha - neuronio.linha)**2) + (bmu.coluna - neuronio.coluna)**2)))
                        taxa_vizinhanca = math.exp(- r/(2*sigma**2))
                        for i in range (len(entrada)):
                            neuronio.w[i] += taxa_vizinhanca * (entrada[i] - neuronio.w[i]) * eta

def treinar(self):
        eta = self.taxa_aprendizado
        for k in range(self.iteracoes):
            for linha in self.matriz:
                for neuronio in linha:
                    neuronio.entradas = []
            sigma = self.sigma_vizinhanca_linear(k) #????
            for i in range(len (self.entradas)):
                bmu = self.eleger_bmu(self.entradas[i])
                self.atualizar_pesos(bmu, eta, sigma, self.entradas[i])

            if self.decaimento == 'linear':
                eta = taxa_aprendizado * (1 - (k/self.iteracoes))
            else:
                eta = self.taxa_aprendizado * math.exp(- (k/(self.iteracoes/2)))
            
