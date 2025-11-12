"""
Script 6: Funciones
===================
Este script enseña:
- Definir funciones
- Parámetros y argumentos
- Valores de retorno
- Parámetros por defecto
- Argumentos variables (*args, **kwargs)
- Scope (alcance) de variables
- Funciones lambda
"""

print("=" * 50)
print("FUNCIONES EN PYTHON")
print("=" * 50)
print()

# 1. FUNCIÓN SIMPLE
print("1. FUNCIÓN SIMPLE")

def saludar():
    """Función que imprime un saludo"""
    print("¡Hola! Bienvenido al programa")

saludar()
print()

# 2. FUNCIÓN CON PARÁMETROS
print("2. FUNCIÓN CON PARÁMETROS")

def saludar_persona(nombre):
    """Función que saluda a una persona específica"""
    print(f"¡Hola {nombre}! ¿Cómo estás?")

saludar_persona("Ana")
saludar_persona("Carlos")
print()

# 3. FUNCIÓN CON RETORNO
print("3. FUNCIÓN CON RETORNO")

def sumar(a, b):
    """Función que suma dos números y retorna el resultado"""
    resultado = a + b
    return resultado

suma1 = sumar(5, 3)
suma2 = sumar(10, 20)
print(f"5 + 3 = {suma1}")
print(f"10 + 20 = {suma2}")
print()

# 4. FUNCIÓN CON MÚLTIPLES PARÁMETROS
print("4. FUNCIÓN CON MÚLTIPLES PARÁMETROS")

def calcular_area_rectangulo(base, altura):
    """Calcula el área de un rectángulo"""
    area = base * altura
    return area

area = calcular_area_rectangulo(5, 10)
print(f"Área del rectángulo (5 × 10): {area}")
print()

# 5. PARÁMETROS POR DEFECTO
print("5. PARÁMETROS POR DEFECTO")

def potencia(base, exponente=2):
    """Calcula la potencia de un número (por defecto al cuadrado)"""
    return base ** exponente

print(f"3^2 = {potencia(3)}")  # Usa exponente por defecto
print(f"2^5 = {potencia(2, 5)}")  # Especifica exponente
print()

# 6. MÚLTIPLES VALORES DE RETORNO
print("6. MÚLTIPLES VALORES DE RETORNO")

def operaciones_basicas(a, b):
    """Realiza varias operaciones y retorna todos los resultados"""
    suma = a + b
    resta = a - b
    multiplicacion = a * b
    division = a / b if b != 0 else None
    return suma, resta, multiplicacion, division

s, r, m, d = operaciones_basicas(10, 5)
print(f"Operaciones con 10 y 5:")
print(f"Suma: {s}, Resta: {r}, Multiplicación: {m}, División: {d}")
print()

# 7. ARGUMENTOS VARIABLES (*args)
print("7. ARGUMENTOS VARIABLES (*args)")

def sumar_todos(*numeros):
    """Suma cualquier cantidad de números"""
    total = sum(numeros)
    return total

print(f"Suma de 1, 2, 3: {sumar_todos(1, 2, 3)}")
print(f"Suma de 5, 10, 15, 20: {sumar_todos(5, 10, 15, 20)}")
print()

# 8. ARGUMENTOS CON NOMBRE (**kwargs)
print("8. ARGUMENTOS CON NOMBRE (**kwargs)")

def mostrar_info(**datos):
    """Muestra información usando argumentos con nombre"""
    for clave, valor in datos.items():
        print(f"{clave}: {valor}")

print("Información del estudiante:")
mostrar_info(nombre="Juan", edad=20, carrera="Ingeniería", promedio=8.5)
print()

# 9. SCOPE DE VARIABLES
print("9. SCOPE DE VARIABLES")

variable_global = "Soy global"

def funcion_scope():
    variable_local = "Soy local"
    print(f"Dentro de la función: {variable_global}")
    print(f"Dentro de la función: {variable_local}")

funcion_scope()
print(f"Fuera de la función: {variable_global}")
# print(variable_local)  # Esto daría error
print()

# 10. FUNCIONES LAMBDA
print("10. FUNCIONES LAMBDA")

# Función lambda simple
cuadrado = lambda x: x ** 2
print(f"Cuadrado de 5: {cuadrado(5)}")

# Lambda con múltiples parámetros
suma = lambda a, b: a + b
print(f"Suma de 3 y 7: {suma(3, 7)}")

# Usar lambda con map
numeros = [1, 2, 3, 4, 5]
cubos = list(map(lambda x: x ** 3, numeros))
print(f"Cubos de {numeros}: {cubos}")
print()

# 11. EJERCICIO PRÁCTICO: CALCULADORA CON FUNCIONES
print("11. EJERCICIO: CALCULADORA CON FUNCIONES")

def sumar_calc(a, b):
    return a + b

def restar_calc(a, b):
    return a - b

def multiplicar_calc(a, b):
    return a * b

def dividir_calc(a, b):
    if b != 0:
        return a / b
    else:
        return "Error: División por cero"

def calculadora():
    """Calculadora interactiva"""
    print("\n--- CALCULADORA ---")
    print("1. Sumar")
    print("2. Restar")
    print("3. Multiplicar")
    print("4. Dividir")
    
    opcion = input("Elige una operación (1-4): ")
    
    if opcion in ['1', '2', '3', '4']:
        num1 = float(input("Primer número: "))
        num2 = float(input("Segundo número: "))
        
        if opcion == '1':
            print(f"Resultado: {sumar_calc(num1, num2)}")
        elif opcion == '2':
            print(f"Resultado: {restar_calc(num1, num2)}")
        elif opcion == '3':
            print(f"Resultado: {multiplicar_calc(num1, num2)}")
        elif opcion == '4':
            print(f"Resultado: {dividir_calc(num1, num2)}")
    else:
        print("Opción inválida")

calculadora()
print()

# 12. DOCUMENTACIÓN DE FUNCIONES
print("12. DOCUMENTACIÓN DE FUNCIONES")

def area_circulo(radio):
    """
    Calcula el área de un círculo.
    
    Parámetros:
        radio (float): El radio del círculo
    
    Retorna:
        float: El área del círculo
    """
    import math
    return math.pi * radio ** 2

print(f"Área de un círculo con radio 5: {area_circulo(5):.2f}")
print(f"\nDocumentación de la función:")
print(area_circulo.__doc__)
print()

print("=" * 50)
print("Fin del programa de funciones")
print("=" * 50)

