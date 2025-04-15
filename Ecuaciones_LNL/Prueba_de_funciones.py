import numpy as np
import LinearEquations as LE;
import NonlinearEquations as NE;

# Sistema: 
# 2x + y - z = 8
# -3x - y + 2z = -11
# -2x + y + 2z = -3
matriz = [
    [2, 1, -1, 8],
    [-3, -1, 2, -11],
    [-2, 1, 2, -3]
]
soluciones1 = LE.eliminacion_Gauss(matriz)
print("Solución por Eliminación de Gauss:", soluciones1)  # Debería dar [2, 3, -1]

soluciones2 = LE.Gauss_jordan(matriz)
print("Matriz resultante de Gauss-Jordan:")
print(np.array(soluciones2))  # La última columna contiene las soluciones [2, 3, -1]

# Matriz 3x3
matriz_determinante = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
det = LE.calcular_determinante(matriz_determinante)
print("Determinante:", det)  # Debería dar 0

# Sistema:
# 3x + 2y = 7
# -x + y = -1
coeficientes = [
    [3, 2],
    [-1, 1]
]
resultados = [7, -1]
soluciones = LE.Crammer(coeficientes, resultados)
print("Solución por Cramer:", soluciones)  # Debería dar [1.8, 0.8]

# Sistema:
# x + 2y + 3z = 1
# 4x + 5y + 6z = 2
# 7x + 8y + 9z = 3
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
B = np.array([1, 2, 3])
solucion = LE.Descomposición_LU(A, B)
print("Solución por LU:", solucion)  # Nota: Esta matriz es singular, podría dar error

# Sistema:
# 4x + y = 9
# x + 3y = 7
A1 = np.array([
    [4, 1],
    [1, 3]
])
b1 = np.array([9, 7])
x0 = np.array([0.0, 0.0])  # Estimación inicial
tol = 1e-6
max_iter = 100
solucion, iteraciones = LE.metodo_jacobi(A1, b1, x0, tol, max_iter)
print(f"Solución por Jacobi: {solucion} en {iteraciones} iteraciones")  # Debería aproximarse a [2, 1.666...]

x01 = np.array([0.0, 0.0])  # Estimación inicial
tol = 1e-6
max_iter = 100
solucion, iteraciones = LE.metodo_gauss_seidel(A1, b1, x01, tol, max_iter)
print(f"Solución por Gauss-Seidel: {solucion} en {iteraciones} iteraciones")  # Debería aproximarse a [2, 1.666...]

# Ejemplo de uso:
def funcion(x):
    return x**3 - x - 2  # Ejemplo de función no lineal

a = 1  # Límite inferior del intervalo
b = 2  # Límite superior del intervalo


try:
    raiz, iteraciones = NE.metodo_biseccion(funcion, a, b, tol, max_iter)
    print(f"La raíz aproximada es: {raiz}")
    print(f"Iteraciones realizadas: {iteraciones}")
except ValueError as e:
    print(e)