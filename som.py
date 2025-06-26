from neuronio.py import neuronio

"""
entradas =[] #usar como entrada para o SOM
for data in dataset:
  entrada= normalizar_entrada(data_dados, data_max_min)
  entradas.append(entrada)

def normalizar_entrada(dados,max_min):
  dados_normalizados = []
  for i in range len(dados): 
    x = (dados[i] - max_min[i][0])/(max_min[i][1] -max_min[i][0])
    dados_normalizados.append(x)
  return dados_normalizados

class SOM:

 def __init__(self, entradas, linhas, colunas, iteracoes, taxa_aprendizado, decaimento='linear'):
        self.iteracoes = iteracoes
        self.entradas = entradas #entradas já normalizadas
        self.taxa_aprendizado = taxa_aprendizado
        self.matriz = []
        self.linhas = linhas
        self.colunas = colunas
        self.decaimento = decaimento
        self.sigma_vizinhanca = 1  #????
        for i in range(linhas):
            self.matriz.append([])  #lista onde o neurônio será adicionado
            for j in range(colunas):
                self.matriz[i].append (Neuronio(numero_atributos=len(entradas[0]),linha=i, coluna=j, pesos_aleatorios=True))

  def calculo_bmu(self,entrada):  #Seleciona qual o neurônio correspondnete para entrada 
    bmu = None
    d_bmu = 0
    for l in range(self.linhas):
      for c in range(self.colunas):
        neuronio = self.matriz[l][c]
        n = len(self.entrada)
        dj = self.distancia_eucliana(n, entrada, neuronio.w) #com a atualização dos pesos as entradas vão sendo relacionadas a novos neuronios
        if bmu is None or dj < d_bmu:
            d_bmu = dj
            bmu = neuronio
      bmu.entradas.append(entrada)
      return bmu
      
def distancia_euclidiana(self,n, a, b): #caclula a distância euclidiana entre dois vetores de tamanho n 
  sum = 0
  for i in range(n):
    sum += (a[i]-b[i])**2
  return math.sqrt(sum)
  
def ajuste_pesos(self,entrada bmu, eta, sigma): #para cada neuronio, atualiza seu pesos
        for linha in self.matriz: 
                for neuronio in linha:
                        r = self.distancia_euclidiana(2, [bmu.linha, bmu.coluna], [neuronio.linha, neuronio.coluna]) #distancia entre neuronio e bmu, quanto mais longe menos aprende
                        #r = math.sqrt(((bmu.linha - neuronio.linha)**2) + (bmu.coluna - neuronio.coluna)**2)))
                        taxa_vizinhanca = math.exp(- r/(2*sigma**2))  #quanto maior o sigma, maior a influencia no aprendizado
                        for i in range (len(entrada)):
                            neuronio.w[i] += taxa_vizinhanca * (entrada[i] - neuronio.w[i]) * eta 

def treinar(self):  
        eta = self.taxa_aprendizado
        for k in range(self.iteracoes):
            for linha in self.matriz:  # para cada neurônio no grid
                for neuronio in linha:
                    neuronio.entradas = []  #limpamos a entrada para selecionar novo bmu
            sigma = self.sigma_vizinhanca_linear(k) #????
            for i in range(len (self.entradas)): #para cada entrada 
                bmu = self.eleger_bmu(self.entradas[i])  #selecionamos um bmu 
                self.atualizar_pesos(bmu, eta, sigma, self.entradas[i])  #atualizmos os pesos de todos os neuronios do grid 

            if self.decaimento == 'linear':
                eta = taxa_aprendizado * (1 - (k/self.iteracoes))
            else:
                eta = self.taxa_aprendizado * math.exp(- (k/(self.iteracoes/2)))
            
