"""
Script 7: Cadenas de Texto (Strings)
=====================================
Este script enseña:
- Crear y manipular strings
- Métodos de strings
- Formateo de strings
- Slicing (rebanado)
- Operaciones comunes
"""

print("=" * 50)
print("CADENAS DE TEXTO EN PYTHON")
print("=" * 50)
print()

# 1. CREAR STRINGS
print("1. CREAR STRINGS")
texto1 = "Hola Mundo"
texto2 = 'Python es genial'
texto3 = """Este es un texto
de múltiples
líneas"""
print(f"Texto 1: {texto1}")
print(f"Texto 2: {texto2}")
print(f"Texto 3:\n{texto3}")
print()

# 2. ACCEDER A CARACTERES
print("2. ACCEDER A CARACTERES")
palabra = "Python"
print(f"Palabra: {palabra}")
print(f"Primer carácter: {palabra[0]}")
print(f"Último carácter: {palabra[-1]}")
print(f"Primeros 3 caracteres: {palabra[0:3]}")
print()

# 3. MÉTODOS DE STRINGS
print("3. MÉTODOS DE STRINGS")
texto = "hola mundo python"
print(f"Original: {texto}")
print(f"Mayúsculas: {texto.upper()}")
print(f"Minúsculas: {texto.lower()}")
print(f"Capitalizado: {texto.capitalize()}")
print(f"Título: {texto.title()}")
print()

# 4. BÚSQUEDA Y REEMPLAZO
print("4. BÚSQUEDA Y REEMPLAZO")
frase = "Python es un lenguaje de programación"
print(f"Frase: {frase}")
print(f"¿Contiene 'Python'?: {'Python' in frase}")
print(f"Posición de 'lenguaje': {frase.find('lenguaje')}")
print(f"Reemplazar 'Python' por 'Java': {frase.replace('Python', 'Java')}")
print()

# 5. DIVIDIR Y UNIR
print("5. DIVIDIR Y UNIR STRINGS")
texto = "manzana,banana,naranja,uva"
frutas = texto.split(",")
print(f"Texto original: {texto}")
print(f"Lista dividida: {frutas}")

nueva_frase = " - ".join(frutas)
print(f"Texto unido: {nueva_frase}")
print()

# 6. FORMATEO DE STRINGS
print("6. FORMATEO DE STRINGS")
nombre = "Ana"
edad = 25
altura = 1.65

# Método 1: f-strings (recomendado)
print(f"Hola, soy {nombre}, tengo {edad} años y mido {altura}m")

# Método 2: format()
print("Hola, soy {}, tengo {} años".format(nombre, edad))

# Método 3: % (antiguo)
print("Hola, soy %s, tengo %d años" % (nombre, edad))
print()

# 7. OPERACIONES CON STRINGS
print("7. OPERACIONES CON STRINGS")
str1 = "Hola"
str2 = "Mundo"
print(f"Concatenación: {str1 + ' ' + str2}")
print(f"Repetición: {str1 * 3}")
print(f"Longitud: {len(str1)}")
print()

# 8. VALIDACIONES
print("8. VALIDACIONES")
texto_num = "12345"
texto_alpha = "Python"
texto_alnum = "Python3"

print(f"'{texto_num}' es numérico: {texto_num.isdigit()}")
print(f"'{texto_alpha}' es alfabético: {texto_alpha.isalpha()}")
print(f"'{texto_alnum}' es alfanumérico: {texto_alnum.isalnum()}")
print()

# 9. ELIMINAR ESPACIOS
print("9. ELIMINAR ESPACIOS")
texto_espacios = "   Hola Mundo   "
print(f"Original: '{texto_espacios}'")
print(f"strip(): '{texto_espacios.strip()}'")
print(f"lstrip(): '{texto_espacios.lstrip()}'")
print(f"rstrip(): '{texto_espacios.rstrip()}'")
print()

# 10. EJERCICIO PRÁCTICO: ANÁLISIS DE TEXTO
print("10. EJERCICIO: ANÁLISIS DE TEXTO")
texto_usuario = input("Ingresa un texto: ")

print(f"\n--- ANÁLISIS ---")
print(f"Longitud: {len(texto_usuario)} caracteres")
print(f"Palabras: {len(texto_usuario.split())} palabras")
print(f"Mayúsculas: {texto_usuario.upper()}")
print(f"Minúsculas: {texto_usuario.lower()}")
print(f"Primera letra mayúscula: {texto_usuario.capitalize()}")
print(f"Invertido: {texto_usuario[::-1]}")
print()

print("=" * 50)
print("Fin del programa de strings")
print("=" * 50)

