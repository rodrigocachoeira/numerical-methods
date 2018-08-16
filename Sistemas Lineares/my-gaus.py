import numpy as np

A = np.array([
        [-3., 1., -2., -2.],
        [2., 5., 1., 7.],
        [-1., 2., 4., 1.]
    ])


#Método de resolução de equações lineares
#pelo método de gaus
class Gaus:
    
    #Define a Matriz a ser resolvida
    def __init__(self, matrix):
        self.matrix = matrix
        self.storage_matrix = np.copy(matrix)
        self.solution = []
        
    #Prepara a Matriz para a resolução
    def prepare(self):
        pass
    
    #Transforma a matriz por meio de operações matemáticas
    #para que seja possível resolver o problema
    def transform(self):
        for i, row in enumerate(self.matrix): #Percorre todas as linhas
            pivot = row[i] #Define a linha Pivot
            j = i + 1 #Define o número da linha a ser alterada
            for pointer in self.matrix[i + 1:]: #Percorre a desde a linha abaixo do pivot até a última linha
                divisor = pointer[i] / pivot # Calcula LA - C.LP, onde LA = Linha Alterada, C = Constante e LP = Linha Pivot
                self.matrix[j] = pointer - np.array(row * divisor) #Estabiliza a Linha Alterada
                j += 1 #Parte para a próxima linha a ser alterada
                
        return self.matrix #Retorna a matriz escalonada
    
    #Resolve a matrix
    #Obs: essa etapa somente pode ser chamada após o escalonamento ser feito
    def resolution(self):
        i = len(self.matrix) - 1 #index de baixo para cima
        for equation in self.matrix[::-1]: #pega equação por equação a partir do ultima elemento
            #               B       /  SUM (X1 + X2 + X3)   
            xi = round(equation[-1] / np.sum(equation[0:-1]))
            self.matrix[:,i] *= xi #multiplica toda a coluna referente a x1 pelo resultado descoberto
            self.solution.append(xi)
            
        self.solution = self.solution[::-1]
        return self.solution[::-1]
    
    #Valida a solução encontrada com a matriz inicial
    def valid_solution(self):
        A = self.storage_matrix[:3,:3] #Pega apenas a matriz A, ignorar a B
        Ax = A * np.array(self.solution).T #Multiplica a matriz dos coeficientes com a matrix dos elementos ocultos
        
        B = np.array([self.storage_matrix[:,-1]]).T #pega a transposta da matriz B
        AxB = np.hstack((Ax, B)) #Junta a matriz novamente
        
        for equation in AxB:    
            #Os dois lados da equação devem ser o mesmo valor, logo se sua divisão não acarretar em 1, o algoritmo é falho
            if (equation[-1] / np.sum(equation[:-1]) != 1.):
                return False
        return True
                        
gaus = Gaus(A)
obtained_matrix = gaus.transform()
solution = gaus.resolution()

print(gaus.valid_solution())