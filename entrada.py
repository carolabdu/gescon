class Entrada: 

  def __init__(self, dados, max_min):
    self.dados_normalizados = []
    for i in range len(dados): 
      x = (dados[i] - max_min[i][0])/(max_min[i][1] -max_min[i][0])
      self.dados_normalizados.append(x)
