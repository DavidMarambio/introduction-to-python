"""
Script 3: Bucles (Loops) - FOR y WHILE
=======================================
Este script enseña:
- Bucle for con range()
- Bucle for con listas
- Bucle while
- break y continue
- Bucles anidados
"""

print("=" * 50)
print("BUCLES EN PYTHON")
print("=" * 50)
print()

# 1. BUCLE FOR CON RANGE
# El comando range se utiliza para generar una secuencia de enteros. 
# Se usa comúnmente en forbucles para iterar un número específico de veces. 
# La función range puede aceptar tres argumentos: start, stop, y step. 
# start Es opcional y su valor predeterminado es 0, 
# stop es obligatorio y define dónde termina la secuencia (exclusivo), 
# y step es opcional y su valor predeterminado es 1.
print("1. BUCLE FOR - Contar del 1 al 5")
for i in range(1, 6):
    print(f"Número: {i}")
print()

# 2. BUCLE FOR - TABLA DE MULTIPLICAR
print("2. TABLA DE MULTIPLICAR")
numero = int(input("¿Qué tabla de multiplicar quieres ver? "))
for i in range(1, 11):
    resultado = numero * i
    print(f"{numero} × {i} = {resultado}")
print()

# 3. BUCLE FOR CON LISTAS
print("3. BUCLE FOR CON LISTAS")
frutas = ["manzana", "banana", "naranja", "uva", "pera"]
print("Lista de frutas:")
for fruta in frutas:
    print(f"- {fruta}")
print()

# 4. BUCLE WHILE
print("4. BUCLE WHILE - Cuenta regresiva")
contador = 5
while contador > 0:
    print(f"Cuenta regresiva: {contador}")
    contador -= 1
print("¡Despegue! 🚀")
print()

# 5. BUCLE WHILE CON CONDICIÓN
print("5. ADIVINA EL NÚMERO")
numero_secreto = 7
intentos = 0

while True:
    intento = int(input("Adivina el número (1-10): "))
    intentos += 1
    
    if intento == numero_secreto:
        print(f"¡Correcto! Lo adivinaste en {intentos} intentos")
        break
    elif intento < numero_secreto:
        print("Muy bajo, intenta de nuevo")
    else:
        print("Muy alto, intenta de nuevo")
print()

# 6. BREAK Y CONTINUE
print("6. USO DE BREAK Y CONTINUE")
print("Números del 1 al 10, saltando el 5:")
for i in range(1, 11):
    if i == 5:
        continue  # Salta el 5
    if i == 9:
        break  # Termina el bucle en 9
    print(i, end=" ")
print("\n")

# 7. BUCLES ANIDADOS
print("7. BUCLES ANIDADOS - Patrón de asteriscos")
filas = int(input("¿Cuántas filas quieres? "))
for i in range(1, filas + 1):
    for j in range(i):
        print("*", end="")
    print()
print()

# 8. SUMA DE NÚMEROS
print("8. SUMA DE NÚMEROS")
suma = 0
cantidad = int(input("¿Cuántos números quieres sumar? "))
for i in range(cantidad):
    num = float(input(f"Ingresa el número {i+1}: "))
    suma += num
print(f"La suma total es: {suma}")
print()

print("=" * 50)
print("Fin del programa de bucles")
print("=" * 50)

