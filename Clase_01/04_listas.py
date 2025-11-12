"""
Script 4: Listas (Arrays)
==========================
Este script enseña:
- Crear listas
- Acceder a elementos
- Modificar elementos
- Métodos de listas (append, remove, insert, etc.)
- Recorrer listas
- Listas por comprensión
"""

print("=" * 50)
print("LISTAS EN PYTHON")
print("=" * 50)
print()

# 1. CREAR LISTAS
print("1. CREAR LISTAS")
numeros = [1, 2, 3, 4, 5]
frutas = ["manzana", "banana", "naranja"]
mixta = [1, "hola", 3.14, True]

print(f"Lista de números: {numeros}")
print(f"Lista de frutas: {frutas}")
print(f"Lista mixta: {mixta}")
print()

# 2. ACCEDER A ELEMENTOS
print("2. ACCEDER A ELEMENTOS")
print(f"Primera fruta: {frutas[0]}")
print(f"Segunda fruta: {frutas[1]}")
print(f"Última fruta: {frutas[-1]}")
print(f"Primeros 3 números: {numeros[0:3]}")
print()

# 3. MODIFICAR ELEMENTOS
print("3. MODIFICAR ELEMENTOS")
frutas[1] = "fresa"
print(f"Lista modificada: {frutas}")
print()

# 4. MÉTODOS DE LISTAS
print("4. MÉTODOS DE LISTAS")
mi_lista = [10, 20, 30]
print(f"Lista inicial: {mi_lista}")

# Agregar elementos
mi_lista.append(40)
print(f"Después de append(40): {mi_lista}")

mi_lista.insert(1, 15)
print(f"Después de insert(1, 15): {mi_lista}")

# Eliminar elementos
mi_lista.remove(20)
print(f"Después de remove(20): {mi_lista}")

ultimo = mi_lista.pop()
print(f"Después de pop(): {mi_lista}, elemento eliminado: {ultimo}")
print()

# 5. OPERACIONES CON LISTAS
print("5. OPERACIONES CON LISTAS")
lista1 = [1, 2, 3]
lista2 = [4, 5, 6]
lista_combinada = lista1 + lista2
print(f"Concatenación: {lista1} + {lista2} = {lista_combinada}")
print(f"Longitud de la lista: {len(lista_combinada)}")
print(f"Máximo: {max(lista_combinada)}")
print(f"Mínimo: {min(lista_combinada)}")
print(f"Suma: {sum(lista_combinada)}")
print()

# 6. RECORRER LISTAS
print("6. RECORRER LISTAS")
colores = ["rojo", "verde", "azul", "amarillo"]
print("Usando for:")
for color in colores:
    print(f"- {color}")

print("\nUsando for con índice:")
for i in range(len(colores)):
    print(f"{i}: {colores[i]}")

# El comando agrega un contador a un iterable y lo devuelve como un objeto enumerado.
# Sintaxis: enumerate(iterable, start=0)
print("\nUsando enumerate:")
for indice, color in enumerate(colores):
    print(f"Posición {indice}: {color}")
print()

# 7. LISTAS POR COMPRENSIÓN
print("7. LISTAS POR COMPRENSIÓN")
cuadrados = [x**2 for x in range(1, 6)]
print(f"Cuadrados del 1 al 5: {cuadrados}")

pares = [x for x in range(1, 11) if x % 2 == 0]
print(f"Números pares del 1 al 10: {pares}")
print()

# 8. EJERCICIO PRÁCTICO: GESTIÓN DE CALIFICACIONES
print("8. EJERCICIO: GESTIÓN DE CALIFICACIONES")
calificaciones = []
cantidad = int(input("¿Cuántas calificaciones quieres ingresar? "))

for i in range(cantidad):
    nota = float(input(f"Ingresa la calificación {i+1}: "))
    calificaciones.append(nota)

print(f"\nCalificaciones ingresadas: {calificaciones}")
print(f"Promedio: {sum(calificaciones) / len(calificaciones):.2f}")
print(f"Nota más alta: {max(calificaciones)}")
print(f"Nota más baja: {min(calificaciones)}")
print()

print("=" * 50)
print("Fin del programa de listas")
print("=" * 50)

