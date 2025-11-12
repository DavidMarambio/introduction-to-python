"""
Script 8: Manejo de Archivos
=============================
Este script enseña:
- Leer archivos
- Escribir archivos
- Modos de apertura
- Manejo de excepciones con archivos
- Trabajar con archivos CSV
"""

print("=" * 50)
print("MANEJO DE ARCHIVOS EN PYTHON")
print("=" * 50)
print()

# 1. ESCRIBIR EN UN ARCHIVO
print("1. ESCRIBIR EN UN ARCHIVO")
archivo = open("ejemplo.txt", "w")
archivo.write("Hola, este es un archivo de prueba\n")
archivo.write("Segunda línea del archivo\n")
archivo.write("Tercera línea del archivo\n")
archivo.close()
print("Archivo 'ejemplo.txt' creado")
print()

# 2. LEER UN ARCHIVO COMPLETO
print("2. LEER UN ARCHIVO COMPLETO")
archivo = open("ejemplo.txt", "r")
contenido = archivo.read()
archivo.close()
print("Contenido del archivo:")
print(contenido)

# 3. LEER LÍNEA POR LÍNEA
print("3. LEER LÍNEA POR LÍNEA")
archivo = open("ejemplo.txt", "r")
print("Leyendo línea por línea:")
for linea in archivo:
    print(f"- {linea.strip()}")
archivo.close()
print()

# 4. USAR WITH (RECOMENDADO)
print("4. USAR WITH (RECOMENDADO)")
with open("ejemplo.txt", "r") as archivo:
    contenido = archivo.read()
    print("Contenido usando 'with':")
    print(contenido)
# El archivo se cierra automáticamente
print()

# 5. AGREGAR CONTENIDO (APPEND)
print("5. AGREGAR CONTENIDO AL ARCHIVO")
with open("ejemplo.txt", "a") as archivo:
    archivo.write("Cuarta línea agregada\n")
    archivo.write("Quinta línea agregada\n")
print("Contenido agregado")

with open("ejemplo.txt", "r") as archivo:
    print("Contenido actualizado:")
    print(archivo.read())
print()

# 6. LEER LÍNEAS EN UNA LISTA
print("6. LEER LÍNEAS EN UNA LISTA")
with open("ejemplo.txt", "r") as archivo:
    lineas = archivo.readlines()
print(f"Total de líneas: {len(lineas)}")
for i, linea in enumerate(lineas, 1):
    print(f"Línea {i}: {linea.strip()}")
print()

# 7. MANEJO DE EXCEPCIONES
print("7. MANEJO DE EXCEPCIONES")
try:
    with open("archivo_inexistente.txt", "r") as archivo:
        contenido = archivo.read()
except FileNotFoundError:
    print("Error: El archivo no existe")
except Exception as e:
    print(f"Error: {e}")
print()

# 8. EJERCICIO: CREAR UN ARCHIVO CSV
print("8. CREAR Y LEER ARCHIVO CSV")
import csv

# Escribir CSV
with open("estudiantes.csv", "w", newline='') as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow(["Nombre", "Edad", "Promedio"])
    escritor.writerow(["Ana", 20, 9.5])
    escritor.writerow(["Carlos", 22, 8.7])
    escritor.writerow(["María", 21, 9.2])
print("Archivo CSV creado")

# Leer CSV
print("\nContenido del CSV:")
with open("estudiantes.csv", "r") as archivo:
    lector = csv.reader(archivo)
    for fila in lector:
        print(fila)
print()

# 9. EJERCICIO PRÁCTICO: DIARIO PERSONAL
print("9. EJERCICIO: DIARIO PERSONAL")

def agregar_entrada():
    """Agrega una entrada al diario"""
    from datetime import datetime
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entrada = input("Escribe tu entrada: ")
    
    with open("diario.txt", "a") as archivo:
        archivo.write(f"[{fecha}] {entrada}\n")
    print("Entrada guardada")

def leer_diario():
    """Lee todas las entradas del diario"""
    try:
        with open("diario.txt", "r") as archivo:
            print("\n--- MI DIARIO ---")
            print(archivo.read())
    except FileNotFoundError:
        print("No hay entradas en el diario todavía")

while True:
    print("\n--- MENÚ ---")
    print("1. Agregar entrada")
    print("2. Leer diario")
    print("3. Salir")
    
    opcion = input("Elige una opción: ")
    
    if opcion == "1":
        agregar_entrada()
    elif opcion == "2":
        leer_diario()
    elif opcion == "3":
        print("¡Hasta luego!")
        break
    else:
        print("Opción inválida")

print()
print("=" * 50)
print("Fin del programa de archivos")
print("=" * 50)

