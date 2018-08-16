import numpy as np

matrix = np.array([
        [-5., 2., -1., -13.],
        [1., 4., 1., 5.],
        [-2., -1., -4., -16.]
    ])

class Jacobi:
    
    def __init__(self, matrix):
        self.matrix = matrix
        self.storage_matrix = np.copy(matrix)
        self.is_possible = False
        
    
    #Realiza a resolução  de uma equação, no qual
    #A -> Matriz dos valores
    #x -> Matriz das variáveis
    #i -> Elemento a ser recalculado
    def solve_equation(self, A, x, i):
        #Inverso do elemento a ser isolado
        y = 1 / A[i]
        
        #Passa todos os elementos para o outro lado da igualdade
        right_side = (A[:-1] * -1)
        #Remove o elemento que queremos isolar
        better_right_side = np.delete(right_side * x, i)
        #Realiza a somatória dos elementos
        sum = np.sum(better_right_side)
        
        #Multiplica o inverso do elemento da vez pelo resultado da equação + o
        #somatório dos demais elementos, com exceção do elemento da vez
        return y * (A[-1] + sum)
    
    #Realiza o reajuste das variáveis até considerar
    #o erro aceitável ou ultrapassar o limite de épocas
    def fit(self, error = 0.002, epochs = 100):
        self.weights = np.zeros((epochs + 1, 3)) #Inicia os pesos iniciais
        self.store_E = np.zeros((epochs + 1, 3))
        
        self.n_epochs = 0
        E = np.array([1, 1, 1])
        
        for epoch in range(1, epochs + 1):
            #Enquanto há elementos na minha matriz de erro maiores
            #que o erro esperado, continuar
            if len(E[np.where(abs(E) > error)]) > 0:
                #Percorre minhas equações
                for i, equation in enumerate(self.matrix):
                    #adiciona o valor encontrado para próxima época utilizá-lo
                    self.weights[epoch, i] = self.solve_equation(equation, self.weights[epoch - 1], i)
                    
                E = self.weights[epoch] - self.weights[epoch - 1]
                #armazena a lista de erros
                self.store_E[epoch] = E
            else:
                self.n_epochs = epoch
                self.is_possible = True
                
                return self.is_possible #Todos os Erros estão abaixo do desejado
            
        return False #Não Encontrou
    
    #Retorna o resultado encontrado
    def result(self):
        if self.is_possible:
            return self.weights[self.n_epochs - 1]
        
        return np.array([])



jacobi = Jacobi(matrix)
jacobi.fit()

print(jacobi.result())