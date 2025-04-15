# Importaciones desde los submódulos
from .LinearEquations import (
    eliminacion_Gauss,
    Gauss_jordan,
    calcular_determinante,
    Crammer,
    Descomposición_LU,
    metodo_jacobi,
    metodo_gauss_seidel
)

from .NonlinearEquations import (
    metodo_biseccion
)

# Lista de lo que se exporta cuando se usa 'from Ecuaciones_LNL import *'
__all__ = [
    'eliminacion_Gauss',
    'Gauss_jordan',
    'calcular_determinante',
    'Crammer',
    'Descomposición_LU',
    'metodo_jacobi',
    'metodo_gauss_seidel',
    'metodo_biseccion'
]