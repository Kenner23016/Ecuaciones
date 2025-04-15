import numpy as np

def metodo_biseccion(func, a, b, tol, max_iter):
    """
    Resuelve una ecuación no lineal utilizando el método de Bisección.
    
    Parámetros:
        func: Función no lineal a resolver.
        a: Límite inferior del intervalo.
        b: Límite superior del intervalo.
        tol: Tolerancia deseada para la raíz.
        max_iter: Número máximo de iteraciones permitido.
        
    Retorna:
        raíz: Valor aproximado de la raíz.
        k: Número de iteraciones realizadas.
    """
    if func(a) * func(b) >= 0:
        raise ValueError("El método de Bisección requiere que f(a) y f(b) tengan signos opuestos.")
    
    for k in range(max_iter):
        # Calcular el punto medio
        c = (a + b) / 2
        
        # Evaluar la función en el punto medio
        f_c = func(c)
        
        # Comprobación de convergencia
        if np.abs(f_c) < tol or (b - a) / 2 < tol:
            return c, k + 1
        
        # Actualizar el intervalo
        if func(a) * f_c < 0:
            b = c
        else:
            a = c
    
    raise ValueError("El método de Bisección no convergió después del máximo número de iteraciones.")

# Ejemplo de uso:
def funcion(x):
    return x**3 - x - 2  # Ejemplo de función no lineal

a = 1  # Límite inferior del intervalo
b = 2  # Límite superior del intervalo
tol = 1e-6  # Tolerancia deseada
max_iter = 100  # Máximo número de iteraciones

try:
    raiz, iteraciones = metodo_biseccion(funcion, a, b, tol, max_iter)
    print(f"La raíz aproximada es: {raiz}")
    print(f"Iteraciones realizadas: {iteraciones}")
except ValueError as e:
    print(e)