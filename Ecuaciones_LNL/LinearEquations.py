import copy
import numpy as np

from scipy.linalg import lu


#Resuelve el sistema de ecuaciones usando metodo de Gaus
def eliminacion_Gauss(matriz):
    cant_ecuaciones = len(matriz);  

    for i in range(cant_ecuaciones):
        for j in range(i +1, cant_ecuaciones):
            f2 = matriz[j][i] / matriz[i][i];
            for k in range(i, cant_ecuaciones + 1): 
                matriz[j][k] -= f2 * matriz[i][k]

    soluciones = [0] * cant_ecuaciones
    for i in range(cant_ecuaciones - 1, -1, -1):  # Desde la última fila hacia arriba
        suma = sum(matriz[i][j] * soluciones[j] for j in range(i + 1, cant_ecuaciones))
        soluciones[i] = (matriz[i][cant_ecuaciones] - suma) / matriz[i][i]

    return soluciones


def Gauss_jordan(matriz):
    cant = len(matriz);

    for i in range(cant):
        pivote = matriz[i][i]
        matriz[i] = [elemento / pivote for elemento in matriz[i]]
        
        for j in range(cant):
            if i != j:  #no se trabaja en la fila del pivote
                factor = matriz[j][i]  # Factor para hacer cero el elemento
                matriz[j] = [matriz[j][k] - factor * matriz[i][k] for k in range(len(matriz[i]))]

    #creamos segundo triangulo
    for i in range(cant - 1, -1, -1):  # Empezamos en la última fila
        pivote = matriz[i][i]  
        matriz[i] = [elemento / pivote for elemento in matriz[i]]  #

       
        for j in range(i - 1, -1, -1):  # Iteramos por las filas superiores
            factor = matriz[j][i]  # Factor para hacer ceros por encima del pivote
            # Restamos un múltiplo de la fila actual (con pivote) de la fila superior
            matriz[j] = [matriz[j][k] - factor * matriz[i][k] for k in range(len(matriz[i]))]

    respuestas =[0] * cant;
    for i in range(cant):
        respuestas [i]= matriz[i][len(matriz[i])-1]
    return matriz;

def calcular_determinante(matriz):#calcula determinantes matriz 2x2 y 3x3
    if len(matriz) == 2:  # Matriz 2x2
        return (matriz[0][0] * matriz[1][1]) - (matriz[0][1] * matriz[1][0])
    
    elif len(matriz) == 3:  # Matriz 3x3
        a, b, c = matriz[0]
        d, e, f = matriz[1]
        g, h, i = matriz[2]
        return (
            a * (e * i - f * h) -
            b * (d * i - f * g) +
            c * (d * h - e * g)
        )

def Crammer(coeficientes, resultados): #para 2 o 3 ecuaciones
    #respuesta
    respuesta = 0;

    if len(coeficientes) != len(resultados):
        print("El número de coeficientes (ecuaciones), incógnitas y resultados debe ser el mismo.")

    #buscar determinante de los coeficienes 
    determinante_general = 0;
    if len(coeficientes) == 2:
        determinante_general = calcular_determinante(coeficientes)
    elif len(coeficientes) == 3:
        determinante_general = calcular_determinante(coeficientes)
    #creamos matrices modificadas para cada cantidad de valores
    #matrices por variable
    matriz_x= [0];
    matriz_y= [0];
    matriz_z= [0];

    #determinantes por variable
    d_x = 0;
    d_y=0;
    d_Z=0;


    if len(coeficientes) == 2:
        matriz_x  = copy.deepcopy(coeficientes);
        matriz_x[0][0] = resultados[0]
        matriz_x [1][0] = resultados[1]
        d_x = calcular_determinante(matriz_x)

        matriz_y = copy.deepcopy(coeficientes);
        matriz_y[0][1] = resultados[0]
        matriz_y [1][1] = resultados[1]
        d_y = calcular_determinante(matriz_y)

        respuesta = [d_x/determinante_general, d_y/determinante_general]
    elif len(coeficientes) == 3:
        matriz_x  = copy.deepcopy(coeficientes);
        matriz_x[0][0] = resultados[0]
        matriz_x [1][0] = resultados[1]
        matriz_x [2][0] = resultados[2]
        d_x = calcular_determinante(matriz_x);

        matriz_y = copy.deepcopy(coeficientes);
        matriz_y[0][1] = resultados[0]
        matriz_y [1][1] = resultados[1]
        matriz_y [2][1] = resultados[2]
        d_y = calcular_determinante(matriz_y);

        matriz_z = copy.deepcopy(coeficientes);
        matriz_z[0][2] = resultados[0]
        matriz_z [1][2] = resultados[1] 
        matriz_z [2][2] = resultados[2]
        d_z = calcular_determinante(matriz_z);

        respuesta = [
            d_x/determinante_general, 
            d_y/determinante_general, 
            d_z/determinante_general
            ]
        
    return respuesta;


def Descomposición_LU(A, B):
    """
    
        Parámetros:
            A: Matriz de coeficientes (numpy array).
            B: Vector de resultados (numpy array).
        
        Retorna:
            X: Vector solución.
    """
    # Factorización LU
    P, L, U = lu(A)  # P es la matriz de permutación
    
    # Resolver L * Y = P * B
    Y = np.linalg.solve(L, np.dot(P, B))
    
    # Resolver U * X = Y
    X = np.linalg.solve(U, Y)
    
    return X

def metodo_jacobi(A, b, x0, tol, max_iter):
    """
    Resuelve el sistema de ecuaciones lineales A * x = b utilizando el método de Jacobi.
    
    Parámetros:
        A: Matriz de coeficientes (numpy array).
        b: Vector de resultados (numpy array).
        x0: Vector inicial para las iteraciones (numpy array).
        tol: Tolerancia deseada para la solución.
        max_iter: Número máximo de iteraciones permitido.
        
    Retorna:
        x: Vector solución.
        k: Número de iteraciones realizadas.
    """
    n = len(A)  # Tamaño del sistema
    x = x0.copy()  # Vector solución inicial
    for k in range(max_iter):
        x_new = np.zeros_like(x)  # Crear un nuevo vector para las iteraciones
        for i in range(n):
            suma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - suma) / A[i, i]
        
        # Comprobación de la convergencia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1  # Retorna la solución y el número de iteraciones
        
        x = x_new  # Actualizar el vector solución
    
    print("El método de Jacobi no convergió después del máximo número de iteraciones.")


def metodo_gauss_seidel(A, b, x0, tol, max_iter):
    """
    Resuelve el sistema de ecuaciones lineales A * x = b utilizando el método de Gauss-Seidel.
    
    Parámetros:
        A: Matriz de coeficientes (numpy array).
        b: Vector de resultados (numpy array).
        x0: Vector inicial para las iteraciones (numpy array).
        tol: Tolerancia deseada para la solución.
        max_iter: Número máximo de iteraciones permitido.
        
    Retorna:
        x: Vector solución.
        k: Número de iteraciones realizadas.
    """
    n = len(A)  # Tamaño del sistema
    x = x0.copy()  # Copia del vector inicial para no modificarlo
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            suma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - suma) / A[i, i]
        
        # Comprobación de convergencia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1  # Retorna la solución y el número de iteraciones
        
        x = x_new  # Actualizar el vector solución
    
    raise ValueError("El método de Gauss-Seidel no convergió después del máximo número de iteraciones.")

