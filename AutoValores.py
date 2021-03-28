import numpy as np
import scipy.linalg as sci

def potencia_regular(A):
    v = np.ones((A.shape[0],1))
    tolerancia = 1E-10

    lambda_ = 0
    dif = 1

    while dif > tolerancia:
        lambda_velho=lambda_
        x1 = v / np.linalg.norm(v)
        v = A.dot(x1)
        lambda_ = x1.dot(v)
        dif = abs((lambda_-lambda_velho)/lambda_)

        #print('autovetor = \n ',x1)
        #print('autovalor = ',lambda_,'\n')

    return x1,lambda_


def potencia_inverso(A):
    v = np.ones((A.shape[0],1))
    tolerancia = 1E-10
    
    LU,piv = sci.lu_factor(A)               #LU tem o U na parte superior e o L na inferior
    
    lambda_ = 0
    dif = 1

    while dif > tolerancia:
        lambda_velho=lambda_
        x1 = v / np.linalg.norm(v)
        v = sci.lu_solve((LU, piv), x1)
        lambda_ = x1.T.dot(v)
        dif = abs((lambda_-lambda_velho)/lambda_)

        #print('autovetor = \n ',x1)
        #print('autovalor = ',lambda_,'\n')
        
    lambda_ = 1/lambda_
    return x1,lambda_

def potencia_deslocamento(A,desloc):
    A = A - (desloc*np.eye(A.shape[0]))

    vetor,valor = potencia_inverso(A)
    valor += desloc
    
    return vetor,valor

if __name__ == '__main__':
    A1 = np.array([[5,2,1],[2,3,1],[1,1,2]])
    A2 = np.array([[-14,1,-2],[1,-1,1],[-2,1,-11]])
    A3 = np.array([[40,8,4,2,1],[8,30,12,6,2],[4,12,20,1,2],[2,6,1,25,4],[1,2,2,4,5]])
    A = (A1,A2,A3)
    
    #Como já conheço o maior e menor de A1 e A3, posso usá-los
    
    desloc1 = np.linspace(1,6,3)     #linspace(a,b,x)  retorna x números igualmente espaçados no intervalo [a,b]
    desloc2 = np.linspace(-20,-3,3)
    desloc3 = np.linspace(4,50,5)
    deslocs = (desloc1,desloc2,desloc3)
    
    for matriz,desloc in zip(A,deslocs):
        print('\nMatriz\n',matriz)
        for p in desloc:
            print('\nDeslocamento = ',p)
            vetor,valor=potencia_deslocamento(matriz,p)
            print('autovetor = \n',vetor)
            print('autovalor = ',valor)

