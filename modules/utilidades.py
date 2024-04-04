import time
from functools import wraps

def calcular_tiempo_ejecucion(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fin = time.time()
        tiempo_ejecucion = fin - inicio
        print(f"La función {func.__name__} tomó {tiempo_ejecucion} segundos en ejecutarse.")
        return resultado
    return wrapper
