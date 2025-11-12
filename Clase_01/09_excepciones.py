"""
Script 9: Manejo de Excepciones
================================
Este script enseña:
- Try-except básico
- Múltiples excepciones
- Finally
- Raise (lanzar excepciones)
- Excepciones personalizadas
"""

print("=" * 50)
print("MANEJO DE EXCEPCIONES EN PYTHON")
print("=" * 50)
print()

# 1. TRY-EXCEPT BÁSICO
print("1. TRY-EXCEPT BÁSICO")
try:
    numero = int(input("Ingresa un número: "))
    resultado = 10 / numero
    print(f"10 / {numero} = {resultado}")
except:
    print("Error: Ocurrió un problema")
print()

# 2. CAPTURAR EXCEPCIONES ESPECÍFICAS
print("2. EXCEPCIONES ESPECÍFICAS")
try:
    numero = int(input("Ingresa un número: "))
    resultado = 10 / numero
    print(f"10 / {numero} = {resultado}")
except ValueError:
    print("Error: Debes ingresar un número válido")
except ZeroDivisionError:
    print("Error: No se puede dividir por cero")
print()

# 3. CAPTURAR EL MENSAJE DE ERROR
print("3. CAPTURAR EL MENSAJE DE ERROR")
try:
    numero = int(input("Ingresa un número: "))
    resultado = 10 / numero
    print(f"10 / {numero} = {resultado}")
except Exception as e:
    print(f"Error: {type(e).__name__} - {e}")
print()

# 4. ELSE Y FINALLY
print("4. ELSE Y FINALLY")
try:
    numero = int(input("Ingresa un número: "))
    resultado = 10 / numero
except ValueError:
    print("Error: Número inválido")
except ZeroDivisionError:
    print("Error: División por cero")
else:
    print(f"Resultado: {resultado}")
finally:
    print("Este bloque siempre se ejecuta")
print()

# 5. MÚLTIPLES EXCEPCIONES EN UNA LÍNEA
print("5. MÚLTIPLES EXCEPCIONES")
try:
    lista = [1, 2, 3]
    indice = int(input("Ingresa un índice (0-2): "))
    print(f"Elemento: {lista[indice]}")
except (ValueError, IndexError) as e:
    print(f"Error: {e}")
print()

# 6. RAISE - LANZAR EXCEPCIONES
print("6. LANZAR EXCEPCIONES CON RAISE")

def verificar_edad(edad):
    if edad < 0:
        raise ValueError("La edad no puede ser negativa")
    if edad < 18:
        raise Exception("Debes ser mayor de edad")
    return "Acceso permitido"

try:
    edad_usuario = int(input("Ingresa tu edad: "))
    resultado = verificar_edad(edad_usuario)
    print(resultado)
except ValueError as e:
    print(f"Error de valor: {e}")
except Exception as e:
    print(f"Error: {e}")
print()

# 7. EXCEPCIONES PERSONALIZADAS
print("7. EXCEPCIONES PERSONALIZADAS")

class SaldoInsuficienteError(Exception):
    """Excepción personalizada para saldo insuficiente"""
    pass

class CuentaBancaria:
    def __init__(self, saldo):
        self.saldo = saldo
    
    def retirar(self, cantidad):
        if cantidad > self.saldo:
            raise SaldoInsuficienteError(
                f"Saldo insuficiente. Saldo actual: ${self.saldo}"
            )
        self.saldo -= cantidad
        return self.saldo

try:
    cuenta = CuentaBancaria(100)
    print(f"Saldo inicial: ${cuenta.saldo}")
    cantidad = float(input("¿Cuánto deseas retirar? $"))
    nuevo_saldo = cuenta.retirar(cantidad)
    print(f"Retiro exitoso. Nuevo saldo: ${nuevo_saldo}")
except SaldoInsuficienteError as e:
    print(f"Error: {e}")
except ValueError:
    print("Error: Ingresa una cantidad válida")
print()

# 8. EJERCICIO PRÁCTICO: CALCULADORA ROBUSTA
print("8. EJERCICIO: CALCULADORA CON MANEJO DE ERRORES")

def calculadora_segura():
    """Calculadora con manejo completo de excepciones"""
    try:
        print("\n--- CALCULADORA ---")
        num1 = float(input("Primer número: "))
        operador = input("Operador (+, -, *, /): ")
        num2 = float(input("Segundo número: "))
        
        if operador == "+":
            resultado = num1 + num2
        elif operador == "-":
            resultado = num1 - num2
        elif operador == "*":
            resultado = num1 * num2
        elif operador == "/":
            if num2 == 0:
                raise ZeroDivisionError("No se puede dividir por cero")
            resultado = num1 / num2
        else:
            raise ValueError("Operador inválido")
        
        print(f"Resultado: {num1} {operador} {num2} = {resultado}")
        
    except ValueError as e:
        print(f"Error de valor: {e}")
    except ZeroDivisionError as e:
        print(f"Error matemático: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")
    finally:
        print("Gracias por usar la calculadora")

calculadora_segura()
print()

# 9. TIPOS COMUNES DE EXCEPCIONES
print("9. TIPOS COMUNES DE EXCEPCIONES")
print("""
Excepciones comunes en Python:
- ValueError: Valor incorrecto
- TypeError: Tipo de dato incorrecto
- ZeroDivisionError: División por cero
- IndexError: Índice fuera de rango
- KeyError: Clave no encontrada en diccionario
- FileNotFoundError: Archivo no encontrado
- AttributeError: Atributo no existe
- ImportError: Error al importar módulo
""")

print("=" * 50)
print("Fin del programa de excepciones")
print("=" * 50)

