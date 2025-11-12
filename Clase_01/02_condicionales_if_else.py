"""
Script 2: Condicionales IF-ELSE
================================
Este script enseña:
- Estructura if
- Estructura if-else
- Estructura if-elif-else
- Operadores de comparación (>, <, ==, !=, >=, <=)
- Operadores lógicos (and, or, not)
"""

print("=" * 50)
print("CONDICIONALES EN PYTHON")
print("=" * 50)
print()

# 1. IF SIMPLE
print("1. CONDICIONAL IF SIMPLE")
edad = int(input("¿Cuántos años tienes? "))

if edad >= 18:
    print("Eres mayor de edad")
print()

# 2. IF-ELSE
print("2. CONDICIONAL IF-ELSE")
numero = int(input("Ingresa un número: "))

if numero > 0:
    print(f"El número {numero} es POSITIVO")
else:
    print(f"El número {numero} es NEGATIVO o CERO")
print()

# 3. IF-ELIF-ELSE
print("3. CONDICIONAL IF-ELIF-ELSE")
nota = float(input("Ingresa tu nota (0-100): "))

if nota >= 90:
    print("Calificación: A - Excelente")
elif nota >= 80:
    print("Calificación: B - Muy Bueno")
elif nota >= 70:
    print("Calificación: C - Bueno")
elif nota >= 60:
    print("Calificación: D - Suficiente")
else:
    print("Calificación: F - Reprobado")
print()

# 4. OPERADORES DE COMPARACIÓN
print("4. OPERADORES DE COMPARACIÓN")
a = int(input("Ingresa el primer número: "))
b = int(input("Ingresa el segundo número: "))

print(f"\n{a} > {b} es: {a > b}")
print(f"{a} < {b} es: {a < b}")
print(f"{a} == {b} es: {a == b}")
print(f"{a} != {b} es: {a != b}")
print(f"{a} >= {b} es: {a >= b}")
print(f"{a} <= {b} es: {a <= b}")
print()

# 5. OPERADORES LÓGICOS (AND, OR, NOT)
print("5. OPERADORES LÓGICOS")
edad = int(input("Ingresa tu edad: "))
tiene_licencia = input("¿Tienes licencia de conducir? (si/no): ").lower()

if edad >= 18 and tiene_licencia == "si":
    print("Puedes conducir")
elif edad >= 18 and tiene_licencia == "no":
    print("Eres mayor de edad pero necesitas licencia")
else:
    print("No puedes conducir (eres menor de edad)")
print()

# 6. EJEMPLO PRÁCTICO: VERIFICAR PAR O IMPAR
print("6. VERIFICAR SI UN NÚMERO ES PAR O IMPAR")
numero = int(input("Ingresa un número: "))

if numero % 2 == 0:
    print(f"El número {numero} es PAR")
else:
    print(f"El número {numero} es IMPAR")
print()

print("=" * 50)
print("Fin del programa de condicionales")
print("=" * 50)

