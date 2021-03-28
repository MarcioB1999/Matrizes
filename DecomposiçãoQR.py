import numpy as np
np.set_printoptions(precision=5,suppress=True)


def MetodoQR(A,tridiag = False,epsilon = 1E-8):
    n = A.shape[0]
    val=100
    P=np.eye(n)

    if tridiag:
        decomposicao = decomposicaoQRH
    else:
        decomposicao = decomposicaoQR

    A_ant=A[:]
    #print("A=", A)

    while(val>epsilon):
        Q, R = decomposicao(A_ant, n)
        A=R.dot(Q)
        A_ant=A[:]
        P=P.dot(Q)
        val=Soma(A, n)
        print(f'Matriz da iteracao QR = {A} \n')

    aut = A
    return P, aut
        
        
def Soma(A, n):
    soma=0
    for i in range(n):
        for j in range(i):
            soma=soma+(A[i,j])**2
    return soma
            
                
def decomposicaoQR(A, n):
    QT=np.eye(n)
    R_ant=A
    for j in range(n-1):#coluna
        for i in range(j+1, n):#linha
            J=matriz_jacobi(R_ant, i, j, n)
            R=J@(R_ant)
            R_ant=R[:]
            QT=J@(QT)

    return QT.transpose(), R
    

def decomposicaoQRH(A, n):
    QT=np.eye(n)
    R_ant=A
    for j in range(n-1):#coluna
        J=matriz_householder(R_ant, j)
        R=J@(R_ant)
        R_ant=R[:]
        QT=J@(QT)

    return QT.transpose(), R

    
def matriz_jacobi(A,i,j,n):
    e = 10E-6
    J = np.eye(n)
    if abs(A[i,j]) <= e:
        return J
    if abs(A[j,j]) <= e: 
        if A[i,j] < 0:
            theta=np.pi/2
        else:
            theta=-np.pi/2
    else:
        theta = np.arctan(-A[i,j]/A[j,j])
         
    J[i,i] = np.cos(theta)
    J[j,j] = np.cos(theta)
    J[i,j] = np.sin(theta)
    J[j,i] = -np.sin(theta)
    
    return J



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


if __name__ == "__main__":
    n = 5
    A = np.array([40,8,4,2,1,8,30,12,6,2,4,12,20,1,2,2,6,1,25,4,1,2,2,4,5]).reshape(n,n)
        
    print(f"Matriz a ser usada no metodo QR \n {A} \n")
    P, autv= MetodoQR(A,tridiag=False)

    print(f"Matriz diagonal A_barra = {autv} \n")
    print(f'Matriz acumulada P = {P} \n')
    auto_valores = np.diag(autv)
    print('Pares Autovalores e Autovetores \n')
    for i in range(n):
        print(f'{auto_valores[i]:.4f} - {P[:,i]} ')

    print('\nAutovetores e Autovalores de A pelo método de Jacobi(Tarefa 14): \n')
    print('Autovalor = 49.38315 --- Autovetor = [0.69806 0.57987 0.34487 0.22686 0.07784] ')
    print('Autovalor = 31.31147 --- Autovetor = [-0.70728  0.5012   0.32981  0.36252  0.0914]') 
    print('Autovalor = 11.64243 --- Autovetor = [ 0.04133 -0.59192  0.77449  0.15537  0.15475]') 
    print('Autovalor = 23.64808 --- Autovetor = [ 0.10329 -0.24728 -0.39382  0.87054  0.12351]') 
    print('Autovalor = 4.01488 --- Autovetor = [-0.0091   0.03207 -0.13179 -0.18745  0.97282] \n\n') 

    print('Metodo QR recebendo a matriz tridiagonal\n')
    A_tridiag, H = householder(A)
    P, autv= MetodoQR(A_tridiag,tridiag=True)                             #Sinalizando que está mandando uma tridiagonal
    print(f'Matriz P de depois da tridiagonalização\n {P} \n')
    HP = H.dot(P)
    print(f'Autovetores de A pelo por HP\n {HP}')
