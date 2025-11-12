"""
Script 1: Interacción Básica - Entrada y Salida de Datos
=========================================================
Este script enseña:
- Mensajes de salida con print()
- Ingreso de datos con input()
- Conversión de tipos de datos
- Operaciones matemáticas básicas
"""

print("=" * 50)
print("BIENVENIDO AL PROGRAMA DE OPERACIONES MATEMÁTICAS")
print("=" * 50)
print()

# 1. SALIDA DE DATOS - Mostrar mensajes en consola
print("1. MENSAJES DE SALIDA")
print("Hola, este es un mensaje simple")
print("Puedo mostrar números:", 42)
print("Puedo mostrar decimales:", 3.14159)
print()

# 2. ENTRADA DE DATOS - Solicitar información al usuario
print("2. ENTRADA DE DATOS")
nombre = input("¿Cuál es tu nombre? ")
print(f"¡Hola {nombre}! Encantado de conocerte.")
print()

# 3. SUMA DE DOS NÚMEROS
print("3. SUMA DE DOS NÚMEROS")
print("Vamos a sumar dos números")
numero1 = input("Ingresa el primer número: ")
numero2 = input("Ingresa el segundo número: ")

# Convertir texto a números
numero1 = float(numero1)
numero2 = float(numero2)

suma = numero1 + numero2
print(f"La suma de {numero1} + {numero2} = {suma}")
print()

# 4. MULTIPLICACIÓN
print("4. MULTIPLICACIÓN")
num_a = float(input("Ingresa el primer número: "))
num_b = float(input("Ingresa el segundo número: "))
multiplicacion = num_a * num_b
print(f"El resultado de {num_a} × {num_b} = {multiplicacion}")
print()

# 5. CALCULADORA BÁSICA
print("5. CALCULADORA BÁSICA")
x = float(input("Ingresa el primer número: "))
y = float(input("Ingresa el segundo número: "))

print(f"\nResultados:")
print(f"Suma: {x} + {y} = {x + y}")
print(f"Resta: {x} - {y} = {x - y}")
print(f"Multiplicación: {x} × {y} = {x * y}")
print(f"División: {x} ÷ {y} = {x / y}")
print(f"Potencia: {x} ^ {y} = {x ** y}")
print()

print("=" * 50)
print("¡Gracias por usar el programa!")
print("=" * 50)

