"""
Script 5: Diccionarios (Dictionaries)
======================================
Este script enseña:
- Crear diccionarios
- Acceder a valores
- Modificar y agregar elementos
- Métodos de diccionarios
- Recorrer diccionarios
- Diccionarios anidados
"""

print("=" * 50)
print("DICCIONARIOS EN PYTHON")
print("=" * 50)
print()

# 1. CREAR DICCIONARIOS
print("1. CREAR DICCIONARIOS")
estudiante = {
    "nombre": "Juan",
    "edad": 20,
    "carrera": "Ingeniería"
}
print(f"Diccionario estudiante: {estudiante}")
print()

# 2. ACCEDER A VALORES
print("2. ACCEDER A VALORES")
print(f"Nombre: {estudiante['nombre']}")
print(f"Edad: {estudiante['edad']}")
print(f"Carrera: {estudiante.get('carrera')}")
print()

# 3. MODIFICAR Y AGREGAR ELEMENTOS
print("3. MODIFICAR Y AGREGAR ELEMENTOS")
estudiante["edad"] = 21
print(f"Edad modificada: {estudiante['edad']}")

estudiante["promedio"] = 8.5
print(f"Diccionario actualizado: {estudiante}")
print()

# 4. MÉTODOS DE DICCIONARIOS
print("4. MÉTODOS DE DICCIONARIOS")
print(f"Claves: {estudiante.keys()}")
print(f"Valores: {estudiante.values()}")
print(f"Pares clave-valor: {estudiante.items()}")
print()

# 5. RECORRER DICCIONARIOS
print("5. RECORRER DICCIONARIOS")
print("Recorriendo claves y valores:")
for clave, valor in estudiante.items():
    print(f"{clave}: {valor}")
print()

# 6. DICCIONARIOS ANIDADOS
print("6. DICCIONARIOS ANIDADOS")
curso = {
    "nombre": "Introducción a Python",
    "estudiantes": {
        "est1": {"nombre": "Ana", "nota": 9.0},
        "est2": {"nombre": "Carlos", "nota": 8.5},
        "est3": {"nombre": "María", "nota": 9.5}
    }
}
print(f"Curso: {curso['nombre']}")
print("Estudiantes:")
for id_est, datos in curso["estudiantes"].items():
    print(f"  {datos['nombre']}: {datos['nota']}")
print()

# 7. EJERCICIO PRÁCTICO: AGENDA DE CONTACTOS
print("7. EJERCICIO: AGENDA DE CONTACTOS")
agenda = {}

while True:
    print("\n--- MENÚ ---")
    print("1. Agregar contacto")
    print("2. Ver contactos")
    print("3. Buscar contacto")
    print("4. Salir")
    
    opcion = input("Elige una opción: ")
    
    if opcion == "1":
        nombre = input("Nombre: ")
        telefono = input("Teléfono: ")
        email = input("Email: ")
        agenda[nombre] = {"telefono": telefono, "email": email}
        print(f"Contacto {nombre} agregado")
    
    elif opcion == "2":
        if agenda:
            print("\n--- CONTACTOS ---")
            for nombre, datos in agenda.items():
                print(f"{nombre}: {datos['telefono']} - {datos['email']}")
        else:
            print("No hay contactos en la agenda")
    
    elif opcion == "3":
        nombre = input("Nombre a buscar: ")
        if nombre in agenda:
            print(f"Teléfono: {agenda[nombre]['telefono']}")
            print(f"Email: {agenda[nombre]['email']}")
        else:
            print("Contacto no encontrado")
    
    elif opcion == "4":
        print("¡Hasta luego!")
        break
    else:
        print("Opción inválida")

print()
print("=" * 50)
print("Fin del programa de diccionarios")
print("=" * 50)

