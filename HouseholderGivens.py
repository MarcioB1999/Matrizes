import numpy as np
np.set_printoptions(precision=5,suppress=True)


def matriz_householder(A, i):
    n = A.shape[0]
    w,z = np.zeros(n),np.zeros(n)                  #w e w'
    
    
    w[i+1:n] = A[i+1:n,i]
    z[i+1] = np.linalg.norm(w) 
    N = w - z
    N /= np.linalg.norm(N)
    
    H = np.eye(n) - (2*np.outer(N,N))
    #print("Matriz de householder = ",H,'\n')
    return H

def householder(A):
    n = A.shape[0]
    H = np.eye(n)
    A_ant = A
    
    for i in range(n-2):
        H_i = matriz_householder(A_ant, i)
        A = H_i.T.dot(A_ant).dot(H_i)
        A_ant = A
        H = H.dot(H_i)
        
    print('Matriz tridiagonal A_barra = \n',A,'\n')
    print("Matriz acumulada H de householder = \n",H,'\n')
    return A, H

def matriz_jacobi(A,i,j,n):
    epsilon = 10E-15
   
    if abs(A[i,j]) <= epsilon: pass
    
    if abs(A[i,i] - A[j,j]) <= epsilon: 
        theta = np.pi/4
    else:
        theta = (1/2) * np.arctan((-2*A[i,j])/(A[i,i] - A[j,j]))
        
    J = np.eye(n)
    J[i,i] = np.cos(theta)
    J[j,j] = np.cos(theta)
    J[i,j] = np.sin(theta)
    J[j,i] = -np.sin(theta)
    
    return J

def varredura_jacobi(A,n,diag):
    J_nova = np.eye(n)           #Vai conter o produto das matrizes ortogonais J_ij com os autovetores da matriz original
    
    A_velha = A

    for j in range(n-1):
        for i in range(j+1,n):
            J_ij = matriz_jacobi(A_velha,i,j,n)
            A_nova = J_ij.T.dot(A_velha).dot(J_ij)        #Transformação de similaridade do passo ij
            if diag: print('A_nova de dentro da varredura = \n',A_nova,'\n')
            A_velha = A_nova
            J_nova = J_nova.dot(J_ij)
    
    
    print("Matriz da varredura de Jacobi = \n",A_nova,'\n')
    return A_nova,J_nova


def val_diag(A,n):
    #somaDosQuadradosDosTermosAbaixoDaDiagonal
    
    soma = 0
    for i in range(n-1):
        soma += A[i+1:,i].dot(A[i+1:,i])                    #Soma dos quadrados de cada vetor abaixo da diagonal
        
    return soma


def metodo_jacobi(A,n,epsilon,diag=False):
    val = 1
    P = np.eye(n)
    A_velha = A
    #cont = 0
    
    
    while val>epsilon: 
        A_nova,J = varredura_jacobi(A_velha,n,diag)  #Devolve uma matriz que deve se aproximar de uma matriz diagonal
        
        A_velha = A_nova
        
        P = P.dot(J)
        
        
        val = val_diag(A_nova,n)                 #Verificar se a matriz já é diagonal
        #print("val ta dando = ",val,'\n')
        # cont+=1
        # if cont == 5: break
        
    print("Matriz diagonal A_barra com os autovalores = \n",A_nova,'\n')
    autovalores = np.diag(A_nova)               #Copia os elementos da diagonal da matriz
    
    return P,autovalores


if __name__ == '__main__':
    n = 5
    A = np.array([40,8,4,2,1,8,30,12,6,2,4,12,20,1,2,2,6,1,25,4,1,2,2,4,5]).reshape(n,n)
    
    vetores,valores = metodo_jacobi(A,n,10E-6)
    print("Matriz acumulada P dos autovetores = \n",vetores,'\n')
    for i in range(n):
        print(f'Autovalor = {valores[i]:.5f}' + f' --- Autovetor = {vetores[:,i]} \n')
    
    print('Autovetores e Autovalores de A pelo método de Householder(Tarefa 13) = \n')
    print([ 0.0091 ,-0.03207  ,0.13179 ,0.18745,-0.97282],'---',[[4.01488]],'\n\n',
          [-0.04133 ,0.59192 ,-0.77449 ,-0.15537 ,-0.15475] ,'---', [[11.64243]],'\n\n',
          [ 0.10328 ,-0.24728 ,-0.39382  ,0.87054  ,0.12351] ,'---', [[23.64808]],'\n\n',
          [ 0.70729 ,-0.5012 , -0.3298 , -0.36252, -0.0914 ] ,'---', [[31.31147]],'\n\n',
          [-0.69806 ,-0.57987, -0.34487, -0.22686, -0.07784] ,'---', [[49.38315]])
    
    print('----'*20)
    tridiag,H = householder(A)
    vetores_tri,valores_tri = metodo_jacobi(tridiag,n,10E-6,diag=True)
    print('Matriz P da matriz tridiagonal = \n',vetores_tri,'\n')
    print('Autovetores de A fazendo HP = \n',H.dot(vetores_tri),'\n')
    
