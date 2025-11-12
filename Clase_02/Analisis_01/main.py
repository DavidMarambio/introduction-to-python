"""
ANALISIS 1: ANÁLISIS DE DATOS CON CSV
======================================

Esta actividad introduce el manejo de archivos CSV y análisis de datos
en tres niveles progresivos:
1. Estadística Básica
2. Estadística Avanzada
3. Ciencia de Datos

Usaremos datos reales de la empresa BMW.
"""

import csv
import os
from datetime import datetime
from collections import Counter, defaultdict


# =============================================================================
# NIVEL 1: ESTADÍSTICA BÁSICA
# =============================================================================

def leer_csv(nombre_archivo, carpeta="./sample"):
    """
    Lee un archivo CSV y retorna una lista de diccionarios.

    Args:
        nombre_archivo: Ruta del archivo CSV
        carpeta: Carpeta donde se encuentra el archivo (default: "../sample")

    Returns:
        Lista de diccionarios donde cada diccionario es una fila
    """
    datos = []
    ruta = os.path.join(carpeta, nombre_archivo)

    # Intentar diferentes encodings, incluyendo utf-8-sig para manejar BOM
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(ruta, 'r', encoding=encoding) as archivo:
                lector = csv.DictReader(archivo)
                for fila in lector:
                    # Limpiar nombres de columnas (quitar BOM si quedó)
                    fila_limpia = {}
                    for key, value in fila.items():
                        key_limpio = key.replace('\ufeff', '').replace('ï»¿', '')
                        fila_limpia[key_limpio] = value
                    datos.append(fila_limpia)
            return datos
        except UnicodeDecodeError:
            datos = []  # Limpiar datos si falla
            continue

    # Si ningún encoding funciona, intentar con errors='ignore'
    with open(ruta, 'r', encoding='utf-8', errors='ignore') as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            # Limpiar nombres de columnas
            fila_limpia = {}
            for key, value in fila.items():
                key_limpio = key.replace('\ufeff', '').replace('ï»¿', '')
                fila_limpia[key_limpio] = value
            datos.append(fila_limpia)

    return datos


def contar_registros(datos):
    """Cuenta el número total de registros."""
    return len(datos)


def obtener_columnas(datos):
    """Obtiene los nombres de todas las columnas."""
    if datos:
        return list(datos[0].keys())
    return []


def contar_valores_unicos(datos, columna):
    """
    Cuenta cuántos valores únicos hay en una columna.
    
    Args:
        datos: Lista de diccionarios
        columna: Nombre de la columna
        
    Returns:
        Número de valores únicos
    """
    valores = set()
    for fila in datos:
        if columna in fila:
            valores.add(fila[columna])
    return len(valores)


def frecuencia_valores(datos, columna):
    """
    Calcula la frecuencia de cada valor en una columna.

    🎯 OBJETIVO:
    Contar cuántas veces aparece cada valor único en una columna.

    📊 ALGORITMO:
    1. Recorrer todos los registros
    2. Para cada valor encontrado, incrementar su contador
    3. Retornar un diccionario con valor: cantidad

    📈 VALORES POSIBLES:
    - Enteros positivos (≥ 0)
    - Representa el número de ocurrencias de cada valor

    💡 INTERPRETACIÓN:
    - Identifica los valores más comunes (moda)
    - Útil para variables categóricas (región, modelo, color, etc.)
    - Ejemplo: Si "Hybrid" aparece 12,716 veces (25.4%), significa que
      aproximadamente 1 de cada 4 vehículos es híbrido

    📊 APLICACIONES:
    - Encontrar el modelo más vendido
    - Identificar la región con más ventas
    - Determinar el tipo de combustible más popular

    Args:
        datos: Lista de diccionarios
        columna: Nombre de la columna

    Returns:
        Diccionario con valores y sus frecuencias
    """
    contador = Counter()
    for fila in datos:
        if columna in fila:
            valor = fila[columna].strip()
            if valor:  # Ignorar valores vacíos
                contador[valor] += 1
    return dict(contador)


def calcular_promedio(datos, columna):
    """
    Calcula el promedio de una columna numérica.

    🎯 OBJETIVO:
    Obtener la media aritmética de un conjunto de valores numéricos.

    📊 ALGORITMO:
    Promedio = Suma de todos los valores / Cantidad de valores

    📈 VALORES POSIBLES:
    - Cualquier número real (puede ser positivo, negativo o cero)
    - Depende del rango de los datos originales

    💡 INTERPRETACIÓN:
    - El promedio representa el "centro" de los datos
    - Es sensible a valores extremos (outliers)
    - Ejemplo: Si el precio promedio es $75,000, significa que en promedio
      los vehículos cuestan $75,000, pero puede haber muchos más baratos
      y algunos muy caros que elevan el promedio

    ⚠️ LIMITACIONES:
    - No indica la dispersión de los datos
    - Puede ser engañoso si hay valores atípicos
    - No es robusto ante outliers

    Args:
        datos: Lista de diccionarios
        columna: Nombre de la columna numérica

    Returns:
        Promedio de los valores
    """
    valores = []
    for fila in datos:
        if columna in fila:
            try:
                # Limpiar el valor (quitar espacios, comas, etc.)
                valor_limpio = fila[columna].strip().replace(',', '')
                if valor_limpio:
                    valores.append(float(valor_limpio))
            except ValueError:
                continue
    
    if valores:
        return sum(valores) / len(valores)
    return 0


def calcular_suma(datos, columna):
    """Calcula la suma total de una columna numérica."""
    valores = []
    for fila in datos:
        if columna in fila:
            try:
                valor_limpio = fila[columna].strip().replace(',', '')
                if valor_limpio:
                    valores.append(float(valor_limpio))
            except ValueError:
                continue
    
    return sum(valores)


def encontrar_minimo_maximo(datos, columna):
    """
    Encuentra el valor mínimo y máximo de una columna numérica.
    
    Returns:
        Tupla (mínimo, máximo)
    """
    valores = []
    for fila in datos:
        if columna in fila:
            try:
                valor_limpio = fila[columna].strip().replace(',', '')
                if valor_limpio:
                    valores.append(float(valor_limpio))
            except ValueError:
                continue
    
    if valores:
        return (min(valores), max(valores))
    return (0, 0)


# =============================================================================
# NIVEL 2: ESTADÍSTICA AVANZADA
# =============================================================================

def calcular_mediana(datos, columna):
    """
    Calcula la mediana de una columna numérica.

    🎯 OBJETIVO:
    Encontrar el valor central que divide los datos en dos mitades iguales.

    📊 ALGORITMO:
    1. Ordenar todos los valores de menor a mayor
    2. Si hay cantidad impar: tomar el valor del medio
    3. Si hay cantidad par: promediar los dos valores centrales

    📈 VALORES POSIBLES:
    - Siempre está dentro del rango de los datos originales
    - Es un valor real que puede o no existir en el dataset

    💡 INTERPRETACIÓN:
    - La mediana es el punto donde el 50% de los datos son menores y 50% mayores
    - Es ROBUSTA ante valores extremos (outliers)
    - Ejemplo: Si la mediana de precios es $75,011, significa que la mitad
      de los vehículos cuestan menos de $75,011 y la otra mitad más

    📊 COMPARACIÓN CON PROMEDIO:
    - Si Mediana ≈ Promedio → Distribución simétrica
    - Si Mediana < Promedio → Hay valores muy altos que elevan el promedio
    - Si Mediana > Promedio → Hay valores muy bajos que reducen el promedio

    ✅ VENTAJAS:
    - No se ve afectada por valores extremos
    - Mejor representación del "valor típico" en datos asimétricos
    """
    valores = []
    for fila in datos:
        if columna in fila:
            try:
                valor_limpio = fila[columna].strip().replace(',', '')
                if valor_limpio:
                    valores.append(float(valor_limpio))
            except ValueError:
                continue
    
    if not valores:
        return 0
    
    valores_ordenados = sorted(valores)
    n = len(valores_ordenados)
    
    if n % 2 == 0:
        # Si hay cantidad par, promedio de los dos valores centrales
        return (valores_ordenados[n//2 - 1] + valores_ordenados[n//2]) / 2
    else:
        # Si hay cantidad impar, el valor central
        return valores_ordenados[n//2]


def calcular_moda(datos, columna):
    """
    Calcula la moda (valor más frecuente) de una columna.
    
    Returns:
        El valor más frecuente y su frecuencia
    """
    frecuencias = frecuencia_valores(datos, columna)
    if not frecuencias:
        return None, 0
    
    valor_mas_frecuente = max(frecuencias, key=frecuencias.get)
    return valor_mas_frecuente, frecuencias[valor_mas_frecuente]


def calcular_desviacion_estandar(datos, columna):
    """
    Calcula la desviación estándar de una columna numérica.

    🎯 OBJETIVO:
    Medir qué tan dispersos o alejados están los datos respecto al promedio.

    📊 ALGORITMO:
    1. Calcular el promedio de todos los valores
    2. Para cada valor, calcular su diferencia con el promedio
    3. Elevar al cuadrado cada diferencia
    4. Promediar todos los cuadrados
    5. Sacar la raíz cuadrada del resultado (esto es la varianza^0.5)

    📈 VALORES POSIBLES:
    - Siempre ≥ 0 (nunca negativo)
    - Valor 0 = todos los datos son iguales
    - Valores altos = datos muy dispersos
    - Se expresa en las mismas unidades que los datos originales

    💡 INTERPRETACIÓN:
    - Indica la "distancia promedio" de los datos respecto a la media
    - Ejemplo: Si el precio promedio es $75,034 y la desviación estándar
      es $25,997, significa que la mayoría de los precios están entre:
      * $75,034 - $25,997 = $49,037 (límite inferior)
      * $75,034 + $25,997 = $101,031 (límite superior)

    📊 REGLA EMPÍRICA (Distribución Normal):
    - ~68% de los datos están dentro de ±1 desviación estándar
    - ~95% de los datos están dentro de ±2 desviaciones estándar
    - ~99.7% de los datos están dentro de ±3 desviaciones estándar

    🔍 COEFICIENTE DE VARIACIÓN:
    - CV = (Desviación Estándar / Promedio) × 100%
    - CV < 15%: Baja variabilidad (datos homogéneos)
    - CV 15-30%: Variabilidad moderada
    - CV > 30%: Alta variabilidad (datos heterogéneos)
    - Ejemplo: CV = 34.65% indica alta variabilidad en precios
    """
    valores = []
    for fila in datos:
        if columna in fila:
            try:
                valor_limpio = fila[columna].strip().replace(',', '')
                if valor_limpio:
                    valores.append(float(valor_limpio))
            except ValueError:
                continue
    
    if len(valores) < 2:
        return 0
    
    promedio = sum(valores) / len(valores)
    varianza = sum((x - promedio) ** 2 for x in valores) / len(valores)
    return varianza ** 0.5


def calcular_percentil(datos, columna, percentil):
    """
    Calcula un percentil específico de una columna numérica.

    🎯 OBJETIVO:
    Encontrar el valor por debajo del cual se encuentra un porcentaje dado de datos.

    📊 ALGORITMO:
    1. Ordenar todos los valores de menor a mayor
    2. Calcular la posición: (percentil/100) × cantidad_de_datos
    3. Retornar el valor en esa posición

    📈 VALORES POSIBLES:
    - Cualquier valor dentro del rango de los datos
    - Depende del percentil solicitado (0-100)

    💡 INTERPRETACIÓN DE PERCENTILES COMUNES:
    - P25 (Cuartil 1): 25% de los datos son menores a este valor
    - P50 (Mediana): 50% de los datos son menores (igual que la mediana)
    - P75 (Cuartil 3): 75% de los datos son menores a este valor
    - P90: 90% de los datos son menores (solo 10% son mayores)
    - P95: 95% de los datos son menores (solo 5% son mayores)

    📊 EJEMPLO CON PRECIOS:
    - P25 = $52,435: El 25% de los vehículos cuestan menos de $52,435
    - P50 = $75,013: El 50% de los vehículos cuestan menos de $75,013
    - P75 = $97,629: El 75% de los vehículos cuestan menos de $97,629
    - P90 = $111,103: Solo el 10% de los vehículos cuestan más de $111,103

    🔍 RANGO INTERCUARTÍLICO (IQR):
    - IQR = P75 - P25
    - Representa el rango donde está el 50% central de los datos
    - Útil para detectar outliers

    Args:
        percentil: Número entre 0 y 100

    Returns:
        Valor en ese percentil
    """
    valores = []
    for fila in datos:
        if columna in fila:
            try:
                valor_limpio = fila[columna].strip().replace(',', '')
                if valor_limpio:
                    valores.append(float(valor_limpio))
            except ValueError:
                continue
    
    if not valores:
        return 0
    
    valores_ordenados = sorted(valores)
    indice = int(len(valores_ordenados) * percentil / 100)
    return valores_ordenados[min(indice, len(valores_ordenados) - 1)]


def agrupar_por_columna(datos, columna_agrupacion, columna_valor, operacion='suma'):
    """
    Agrupa datos por una columna y aplica una operación sobre otra columna.

    🎯 OBJETIVO:
    Realizar análisis agregados por categorías (similar a GROUP BY en SQL).

    📊 ALGORITMO:
    1. Agrupar todos los valores según la columna de agrupación
    2. Para cada grupo, aplicar la operación especificada
    3. Retornar resultados por grupo

    📈 OPERACIONES DISPONIBLES:
    - 'suma': Suma total de valores en cada grupo
    - 'promedio': Media aritmética por grupo
    - 'contar': Cantidad de elementos en cada grupo
    - 'minimo': Valor mínimo en cada grupo
    - 'maximo': Valor máximo en cada grupo

    💡 INTERPRETACIÓN:
    - Permite comparar diferentes categorías
    - Identifica grupos con mejor/peor desempeño

    📊 EJEMPLOS DE USO:

    1. Volumen total por región (operacion='suma'):
       - Asia: 42,974,277 unidades
       - Europe: 42,555,138 unidades
       → Asia es la región con mayor volumen de ventas

    2. Precio promedio por modelo (operacion='promedio'):
       - 7 Series: $75,570 (modelo más caro en promedio)
       - M5: $74,474 (modelo más económico en promedio)
       → Diferencia de ~$1,100 entre modelos

    3. Precio promedio por transmisión:
       - Automatic: $75,171
       - Manual: $74,899
       → Diferencia de solo $272, prácticamente igual

    🎯 APLICACIONES PRÁCTICAS:
    - Comparar ventas entre regiones
    - Analizar precios por categoría de producto
    - Identificar segmentos más rentables
    - Evaluar desempeño por período

    Args:
        columna_agrupacion: Columna por la cual agrupar
        columna_valor: Columna numérica sobre la cual operar
        operacion: 'suma', 'promedio', 'contar', 'minimo', 'maximo'

    Returns:
        Diccionario con grupos y sus valores calculados
    """
    grupos = defaultdict(list)
    
    # Agrupar valores
    for fila in datos:
        if columna_agrupacion in fila and columna_valor in fila:
            grupo = fila[columna_agrupacion].strip()
            try:
                valor_limpio = fila[columna_valor].strip().replace(',', '')
                if valor_limpio:
                    grupos[grupo].append(float(valor_limpio))
            except ValueError:
                continue
    
    # Aplicar operación
    resultado = {}
    for grupo, valores in grupos.items():
        if not valores:
            continue
            
        if operacion == 'suma':
            resultado[grupo] = sum(valores)
        elif operacion == 'promedio':
            resultado[grupo] = sum(valores) / len(valores)
        elif operacion == 'contar':
            resultado[grupo] = len(valores)
        elif operacion == 'minimo':
            resultado[grupo] = min(valores)
        elif operacion == 'maximo':
            resultado[grupo] = max(valores)
    
    return resultado


def filtrar_datos(datos, columna, condicion):
    """
    Filtra datos basándose en una condición.

    Args:
        columna: Nombre de la columna
        condicion: Función que retorna True/False para cada valor

    Returns:
        Lista de filas que cumplen la condición
    """
    resultado = []
    for fila in datos:
        if columna in fila:
            if condicion(fila[columna]):
                resultado.append(fila)
    return resultado


# =============================================================================
# NIVEL 3: CIENCIA DE DATOS
# =============================================================================

def correlacion_categorica(datos, columna1, columna2):
    """
    Analiza la relación entre dos columnas categóricas.

    🎯 OBJETIVO:
    Estudiar cómo se relacionan dos variables categóricas mediante una tabla cruzada.

    📊 ALGORITMO:
    1. Crear una tabla de contingencia (tabla cruzada)
    2. Contar cuántas veces aparece cada combinación de valores
    3. Retornar matriz de frecuencias

    📈 VALORES POSIBLES:
    - Enteros positivos (≥ 0)
    - Representa el conteo de cada combinación

    💡 INTERPRETACIÓN:
    - Muestra la distribución conjunta de dos variables
    - Permite identificar patrones y asociaciones
    - Similar a una tabla dinámica en Excel

    📊 EJEMPLO: Tipo de Combustible × Clasificación de Ventas

    Diesel:
      - Low: 8,505 ventas (69.4% de diesel)
      - High: 3,758 ventas (30.6% de diesel)

    Electric:
      - Low: 8,677 ventas (69.6% de eléctricos)
      - High: 3,794 ventas (30.4% de eléctricos)

    Hybrid:
      - Low: 8,837 ventas (69.5% de híbridos)
      - High: 3,879 ventas (30.5% de híbridos)

    Petrol:
      - Low: 8,735 ventas (69.6% de gasolina)
      - High: 3,815 ventas (30.4% de gasolina)

    🔍 CONCLUSIÓN DEL EJEMPLO:
    - Todos los tipos de combustible tienen distribución similar (~70% Low, ~30% High)
    - NO hay una relación fuerte entre tipo de combustible y clasificación
    - El tipo de combustible NO determina si las ventas serán altas o bajas

    🎯 APLICACIONES:
    - Analizar preferencias por región
    - Estudiar relación entre categorías de productos
    - Identificar combinaciones más comunes

    Returns:
        Diccionario anidado con las frecuencias de cada combinación
    """
    tabla = defaultdict(lambda: defaultdict(int))

    for fila in datos:
        if columna1 in fila and columna2 in fila:
            val1 = fila[columna1].strip()
            val2 = fila[columna2].strip()
            if val1 and val2:
                tabla[val1][val2] += 1

    return dict(tabla)


def analisis_temporal(datos, columna_fecha, columna_valor, formato_fecha='%Y-%m-%dT%H:%M:%S'):
    """
    Analiza tendencias temporales en los datos.

    Args:
        columna_fecha: Columna con fechas
        columna_valor: Columna numérica a analizar
        formato_fecha: Formato de la fecha en el CSV

    Returns:
        Diccionario con fechas y valores agregados
    """
    series_temporal = defaultdict(list)

    for fila in datos:
        if columna_fecha in fila and columna_valor in fila:
            try:
                fecha_str = fila[columna_fecha].strip()
                if not fecha_str:
                    continue

                fecha = datetime.strptime(fecha_str, formato_fecha)
                fecha_clave = fecha.strftime('%Y-%m')  # Agrupar por mes

                valor_limpio = fila[columna_valor].strip().replace(',', '')
                if valor_limpio:
                    series_temporal[fecha_clave].append(float(valor_limpio))
            except (ValueError, KeyError):
                continue

    # Calcular promedio por período
    resultado = {}
    for fecha, valores in sorted(series_temporal.items()):
        resultado[fecha] = {
            'promedio': sum(valores) / len(valores),
            'total': sum(valores),
            'cantidad': len(valores),
            'minimo': min(valores),
            'maximo': max(valores)
        }

    return resultado


def detectar_outliers(datos, columna):
    """
    Detecta valores atípicos (outliers) usando el método IQR.

    🎯 OBJETIVO:
    Identificar valores que son inusualmente altos o bajos comparados con el resto.

    📊 ALGORITMO (Método IQR - Rango Intercuartílico):
    1. Calcular Q1 (percentil 25) y Q3 (percentil 75)
    2. Calcular IQR = Q3 - Q1
    3. Límite inferior = Q1 - 1.5 × IQR
    4. Límite superior = Q3 + 1.5 × IQR
    5. Outliers = valores fuera de estos límites

    📈 VALORES POSIBLES:
    - Cantidad de outliers: 0 a N (total de datos)
    - Límites: Dependen de la distribución de los datos

    💡 INTERPRETACIÓN:
    - Outliers son valores "anormales" o "extremos"
    - Pueden ser errores de medición o casos especiales legítimos
    - Ejemplo: Si el rango normal es $-15,356 a $165,420 y no hay outliers,
      significa que todos los precios están dentro de lo esperado

    🔍 ¿POR QUÉ 1.5 × IQR?
    - Es una regla estándar en estadística (regla de Tukey)
    - Identifica aproximadamente el 0.7% de valores más extremos
    - En distribución normal, captura ~99.3% de los datos como "normales"

    ⚠️ INTERPRETACIÓN DE RESULTADOS:
    - 0 outliers: Datos muy consistentes, sin valores extremos
    - Pocos outliers (<5%): Normal, pueden ser casos especiales
    - Muchos outliers (>10%): Revisar calidad de datos o distribución

    📊 EJEMPLO:
    - Si detectamos 0 outliers en precios, significa que todos los precios
      están dentro del rango esperado y no hay vehículos con precios anormales

    Returns:
        Diccionario con información sobre outliers
    """
    valores = []
    for fila in datos:
        if columna in fila:
            try:
                valor_limpio = fila[columna].strip().replace(',', '')
                if valor_limpio:
                    valores.append(float(valor_limpio))
            except ValueError:
                continue

    if len(valores) < 4:
        return {'outliers': [], 'cantidad': 0}

    valores_ordenados = sorted(valores)
    n = len(valores_ordenados)

    # Calcular cuartiles
    q1 = valores_ordenados[n // 4]
    q3 = valores_ordenados[3 * n // 4]
    iqr = q3 - q1

    # Límites para outliers
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    outliers = [v for v in valores if v < limite_inferior or v > limite_superior]

    return {
        'outliers': outliers,
        'cantidad': len(outliers),
        'limite_inferior': limite_inferior,
        'limite_superior': limite_superior,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    }


def crear_segmentos(datos, columna, num_segmentos=3):
    """
    Divide los datos en segmentos (bins) para análisis.

    🎯 OBJETIVO:
    Convertir datos numéricos continuos en categorías discretas.

    📊 ALGORITMO:
    1. Encontrar el valor mínimo y máximo
    2. Dividir el rango en N partes iguales
    3. Asignar cada valor a su segmento correspondiente

    📈 VALORES POSIBLES:
    - Segmentos: 0, 1, 2, ..., (num_segmentos - 1)
    - Cada segmento representa un rango de valores

    💡 INTERPRETACIÓN:
    - Útil para categorizar datos continuos
    - Facilita el análisis de distribuciones
    - Permite identificar patrones por rangos

    📊 EJEMPLO CON PRECIOS (3 segmentos):
    - Segmento 0 (Económico): 16,715 vehículos
      * Rango: $30,000 - $60,000 aproximadamente
    - Segmento 1 (Medio): 16,647 vehículos
      * Rango: $60,000 - $90,000 aproximadamente
    - Segmento 2 (Premium): 16,638 vehículos
      * Rango: $90,000 - $120,000 aproximadamente

    🎯 APLICACIONES:
    - Segmentación de clientes por valor
    - Clasificación de productos por precio
    - Análisis de distribución de ingresos
    - Categorización de edades, salarios, etc.

    Args:
        num_segmentos: Número de segmentos a crear

    Returns:
        Lista de tuplas (fila, segmento)
    """
    valores = []
    for fila in datos:
        if columna in fila:
            try:
                valor_limpio = fila[columna].strip().replace(',', '')
                if valor_limpio:
                    valores.append((fila, float(valor_limpio)))
            except ValueError:
                continue

    if not valores:
        return []

    # Encontrar min y max
    valores_numericos = [v[1] for v in valores]
    minimo = min(valores_numericos)
    maximo = max(valores_numericos)

    # Calcular tamaño de cada segmento
    rango = (maximo - minimo) / num_segmentos

    # Asignar segmentos
    resultado = []
    for fila, valor in valores:
        if valor == maximo:
            segmento = num_segmentos - 1
        else:
            segmento = int((valor - minimo) / rango)
        resultado.append((fila, segmento))

    return resultado


def analisis_abc(datos, columna_item, columna_valor):
    """
    Realiza un análisis ABC (Pareto) de los datos.

    🎯 OBJETIVO:
    Clasificar elementos según su importancia relativa (Principio de Pareto 80/20).

    📊 ALGORITMO:
    1. Sumar el valor total de cada item
    2. Ordenar items de mayor a menor valor
    3. Calcular porcentaje acumulado
    4. Clasificar:
       - Categoría A: Items que suman el 80% del valor total
       - Categoría B: Items que suman el siguiente 15% (80-95%)
       - Categoría C: Items que suman el último 5% (95-100%)

    📈 VALORES POSIBLES:
    - Categorías: A, B, o C
    - Porcentaje individual: 0% a 100%
    - Porcentaje acumulado: 0% a 100%

    💡 INTERPRETACIÓN (Principio de Pareto):
    - Categoría A: Los "pocos vitales" - Alta prioridad
      * Pocos items que generan la mayor parte del valor
      * Requieren máxima atención y recursos
    - Categoría B: Los "importantes" - Prioridad media
      * Items de importancia moderada
    - Categoría C: Los "muchos triviales" - Baja prioridad
      * Muchos items que generan poco valor
      * Pueden requerir menos atención

    📊 EJEMPLO CON MODELOS BMW:
    - Categoría A: 8 modelos (73% de modelos) generan 80% de las ventas
      * Estos son los modelos estrella que impulsan el negocio
      * Ejemplo: 7 Series, i8, X1, 3 Series, i3
    - Categoría B: 2 modelos generan 15% de las ventas
    - Categoría C: 1 modelo genera solo 5% de las ventas

    🎯 APLICACIONES PRÁCTICAS:
    - Gestión de inventario: Priorizar stock de categoría A
    - Marketing: Enfocar campañas en productos A
    - Producción: Optimizar procesos para items A
    - Ventas: Capacitar equipo en productos A

    Returns:
        Diccionario con items y su clasificación ABC
    """
    # Agrupar por item
    totales = agrupar_por_columna(datos, columna_item, columna_valor, 'suma')

    if not totales:
        return {}

    # Ordenar por valor descendente
    items_ordenados = sorted(totales.items(), key=lambda x: x[1], reverse=True)

    # Calcular total general
    total_general = sum(totales.values())

    # Clasificar
    resultado = {}
    acumulado = 0

    for item, valor in items_ordenados:
        acumulado += valor
        porcentaje_acumulado = (acumulado / total_general) * 100

        if porcentaje_acumulado <= 80:
            categoria = 'A'
        elif porcentaje_acumulado <= 95:
            categoria = 'B'
        else:
            categoria = 'C'

        resultado[item] = {
            'valor': valor,
            'categoria': categoria,
            'porcentaje_individual': (valor / total_general) * 100,
            'porcentaje_acumulado': porcentaje_acumulado
        }

    return resultado


def cruzar_datasets(datos1, datos2, columna_comun):
    """
    Realiza un JOIN entre dos datasets basándose en una columna común.
    Similar a un INNER JOIN en SQL.

    Returns:
        Lista de diccionarios con datos combinados
    """
    # Crear índice del segundo dataset
    indice = defaultdict(list)
    for fila in datos2:
        if columna_comun in fila:
            clave = fila[columna_comun].strip()
            indice[clave].append(fila)

    # Cruzar datos
    resultado = []
    for fila1 in datos1:
        if columna_comun in fila1:
            clave = fila1[columna_comun].strip()
            if clave in indice:
                for fila2 in indice[clave]:
                    # Combinar ambas filas
                    fila_combinada = {**fila1}
                    for k, v in fila2.items():
                        if k != columna_comun:
                            fila_combinada[f"{k}_2"] = v
                    resultado.append(fila_combinada)

    return resultado


def calcular_tasa_crecimiento(datos, columna_fecha, columna_valor, formato_fecha='%Y-%m-%dT%H:%M:%S'):
    """
    Calcula la tasa de crecimiento período a período.

    🎯 OBJETIVO:
    Medir el cambio porcentual de una variable a lo largo del tiempo.

    📊 ALGORITMO:
    1. Agrupar datos por período temporal
    2. Para cada período, calcular:
       Tasa = ((Valor_Actual - Valor_Anterior) / Valor_Anterior) × 100%
    3. Retornar serie temporal con tasas

    📈 VALORES POSIBLES:
    - Tasa > 0: Crecimiento (aumento)
    - Tasa = 0: Sin cambio (estable)
    - Tasa < 0: Decrecimiento (disminución)
    - Expresado en porcentaje (%)

    💡 INTERPRETACIÓN:
    - Muestra la velocidad de cambio entre períodos
    - Permite identificar tendencias y patrones temporales
    - Útil para proyecciones y pronósticos

    📊 EJEMPLO: Crecimiento de Modelos (2010 vs 2024)

    X6: +26.6%
      - 2010: 1,450,874 unidades
      - 2024: 1,836,396 unidades
      - Interpretación: El modelo X6 creció 26.6% en 14 años
      - Es el modelo con mayor crecimiento

    7 Series: +21.5%
      - 2010: 1,388,037 unidades
      - 2024: 1,686,209 unidades
      - Segundo mejor crecimiento

    M5: +2.4%
      - 2010: 1,594,989 unidades
      - 2024: 1,632,996 unidades
      - Crecimiento modesto, casi estable

    🔍 ANÁLISIS:
    - Tasas altas (>20%): Modelos en expansión fuerte
    - Tasas medias (5-20%): Crecimiento moderado
    - Tasas bajas (<5%): Mercado maduro o estancado
    - Tasas negativas: Modelos en declive

    🎯 APLICACIONES:
    - Análisis de tendencias de ventas
    - Evaluación de desempeño de productos
    - Proyección de demanda futura
    - Identificación de oportunidades de crecimiento

    Returns:
        Lista de tuplas (período, valor, tasa_crecimiento)
    """
    series = analisis_temporal(datos, columna_fecha, columna_valor, formato_fecha)

    periodos = sorted(series.keys())
    resultado = []

    for i, periodo in enumerate(periodos):
        valor_actual = series[periodo]['total']

        if i == 0:
            tasa = 0
        else:
            valor_anterior = series[periodos[i-1]]['total']
            if valor_anterior != 0:
                tasa = ((valor_actual - valor_anterior) / valor_anterior) * 100
            else:
                tasa = 0

        resultado.append((periodo, valor_actual, tasa))

    return resultado


def calcular_correlacion_numerica(datos, columna1, columna2):
    """
    Calcula la correlación de Pearson entre dos columnas numéricas.

    🎯 OBJETIVO:
    Medir la fuerza y dirección de la relación lineal entre dos variables.

    📊 ALGORITMO (Correlación de Pearson):
    1. Calcular la media de cada variable
    2. Para cada par de valores, calcular:
       - Desviación de X respecto a su media
       - Desviación de Y respecto a su media
    3. Multiplicar las desviaciones y promediarlas (covarianza)
    4. Dividir por el producto de las desviaciones estándar

    📈 VALORES POSIBLES:
    - Rango: -1 a +1
    - r = +1: Correlación positiva perfecta (línea recta ascendente)
    - r = 0: Sin correlación lineal (no hay relación)
    - r = -1: Correlación negativa perfecta (línea recta descendente)

    💡 INTERPRETACIÓN DE LA MAGNITUD:
    - |r| > 0.7: Correlación FUERTE
      * Las variables están muy relacionadas
    - 0.3 < |r| < 0.7: Correlación MODERADA
      * Hay cierta relación, pero no muy fuerte
    - |r| < 0.3: Correlación DÉBIL o NULA
      * Las variables son prácticamente independientes

    📊 INTERPRETACIÓN DEL SIGNO:
    - r > 0 (Positiva): Cuando una variable aumenta, la otra también
      * Ejemplo: Tamaño de motor vs Precio (más grande = más caro)
    - r < 0 (Negativa): Cuando una variable aumenta, la otra disminuye
      * Ejemplo: Kilometraje vs Precio (más km = más barato)
    - r ≈ 0: No hay relación lineal
      * Ejemplo: Precio vs Kilometraje = -0.0042 (prácticamente 0)
      * Significa que el kilometraje NO afecta el precio de forma lineal

    ⚠️ IMPORTANTE:
    - Correlación NO implica causalidad
    - Solo mide relaciones LINEALES (no detecta relaciones curvas)
    - Sensible a outliers

    📊 EJEMPLO:
    - Precio vs Kilometraje = -0.0042
      * Correlación prácticamente nula
      * El kilometraje no tiene relación lineal con el precio
      * Puede haber otros factores más importantes (modelo, año, etc.)

    Args:
        datos: Lista de diccionarios
        columna1: Nombre de la primera columna numérica
        columna2: Nombre de la segunda columna numérica

    Returns:
        float: Coeficiente de correlación (-1 a 1)
    """
    valores1 = []
    valores2 = []

    for fila in datos:
        try:
            v1 = float(fila.get(columna1, 0))
            v2 = float(fila.get(columna2, 0))
            valores1.append(v1)
            valores2.append(v2)
        except ValueError:
            continue

    if len(valores1) < 2:
        return 0

    # Calcular medias
    media1 = sum(valores1) / len(valores1)
    media2 = sum(valores2) / len(valores2)

    # Calcular covarianza y desviaciones estándar
    covarianza = sum((valores1[i] - media1) * (valores2[i] - media2) for i in range(len(valores1))) / len(valores1)
    std1 = (sum((v - media1) ** 2 for v in valores1) / len(valores1)) ** 0.5
    std2 = (sum((v - media2) ** 2 for v in valores2) / len(valores2)) ** 0.5

    if std1 == 0 or std2 == 0:
        return 0

    return covarianza / (std1 * std2)


def analisis_por_rango(datos, columna, rangos):
    """
    Agrupa datos en rangos personalizados.

    🎯 OBJETIVO:
    Clasificar datos numéricos en categorías definidas por el usuario.

    📊 ALGORITMO:
    1. Definir rangos personalizados con etiquetas
    2. Para cada valor, determinar a qué rango pertenece
    3. Contar cuántos valores caen en cada rango

    📈 VALORES POSIBLES:
    - Enteros positivos (≥ 0)
    - Representa el conteo de elementos en cada rango

    💡 INTERPRETACIÓN:
    - Permite crear segmentaciones personalizadas
    - Útil para análisis de distribución
    - Facilita la comprensión de patrones

    📊 EJEMPLO: Distribución por Rangos de Precio

    Rangos definidos:
    - $30K-$50K: 11,108 vehículos (22.2%)
      * Segmento económico
      * Más de 1 de cada 5 vehículos

    - $50K-$70K: 11,075 vehículos (22.1%)
      * Segmento medio-bajo
      * Distribución muy similar al anterior

    - $70K-$90K: 11,179 vehículos (22.4%)
      * Segmento medio-alto
      * Ligeramente más popular

    - $90K-$110K: 11,015 vehículos (22.0%)
      * Segmento premium
      * Distribución equilibrada

    - $110K+: 5,623 vehículos (11.2%)
      * Segmento lujo
      * Aproximadamente la mitad que otros segmentos

    🔍 ANÁLISIS:
    - Distribución muy equilibrada en los primeros 4 rangos (~22% cada uno)
    - El segmento de lujo ($110K+) tiene la mitad de vehículos
    - Indica un mercado balanceado con enfoque en segmentos medios

    🎯 APLICACIONES:
    - Segmentación de clientes por ingreso
    - Clasificación de productos por precio
    - Análisis de edades por grupos
    - Categorización de ventas por volumen

    Args:
        datos: Lista de diccionarios
        columna: Nombre de la columna numérica
        rangos: Lista de tuplas (min, max, etiqueta)

    Returns:
        dict: Diccionario con conteo por rango
    """
    resultado = {etiqueta: 0 for _, _, etiqueta in rangos}

    for fila in datos:
        try:
            valor = float(fila.get(columna, 0))
            for min_val, max_val, etiqueta in rangos:
                if min_val <= valor < max_val:
                    resultado[etiqueta] += 1
                    break
        except ValueError:
            continue

    return resultado


def calcular_moda_multiple(datos, columna):
    """
    Calcula todas las modas (puede haber múltiples valores con la misma frecuencia máxima).

    Args:
        datos: Lista de diccionarios
        columna: Nombre de la columna

    Returns:
        list: Lista de valores que son moda
    """
    frecuencias = frecuencia_valores(datos, columna)
    if not frecuencias:
        return []

    max_freq = max(frecuencias.values())
    modas = [valor for valor, freq in frecuencias.items() if freq == max_freq]

    return modas


def analisis_varianza(datos, columna):
    """
    Calcula la varianza de una columna numérica.

    🎯 OBJETIVO:
    Medir la dispersión de los datos (qué tan alejados están del promedio).

    📊 ALGORITMO:
    1. Calcular el promedio de todos los valores
    2. Para cada valor, calcular (valor - promedio)²
    3. Promediar todos esos cuadrados

    📈 VALORES POSIBLES:
    - Siempre ≥ 0 (nunca negativo)
    - Valor 0 = todos los datos son iguales
    - Valores altos = datos muy dispersos
    - Se expresa en unidades al cuadrado (ej: USD²)

    💡 INTERPRETACIÓN:
    - La varianza es el cuadrado de la desviación estándar
    - Varianza = (Desviación Estándar)²
    - Ejemplo: Varianza = $675,895,426.74
      * Desviación Estándar = √675,895,426.74 = $25,997.99

    📊 RELACIÓN CON DESVIACIÓN ESTÁNDAR:
    - Varianza: Útil para cálculos matemáticos
    - Desviación Estándar: Más fácil de interpretar (mismas unidades)

    🔍 COEFICIENTE DE VARIACIÓN:
    - CV = (√Varianza / Promedio) × 100%
    - Ejemplo: CV = 34.65% indica alta variabilidad en precios

    Args:
        datos: Lista de diccionarios
        columna: Nombre de la columna numérica

    Returns:
        float: Varianza
    """
    valores = []
    for fila in datos:
        try:
            valor = float(fila.get(columna, 0))
            valores.append(valor)
        except ValueError:
            continue

    if len(valores) < 2:
        return 0

    media = sum(valores) / len(valores)
    varianza = sum((v - media) ** 2 for v in valores) / len(valores)

    return varianza


def top_n_por_metrica(datos, columna_grupo, columna_metrica, n=10, operacion="suma"):
    """
    Obtiene los top N elementos según una métrica.

    🎯 OBJETIVO:
    Identificar los mejores N elementos según un criterio específico.

    📊 ALGORITMO:
    1. Agrupar datos por la columna especificada
    2. Calcular la métrica para cada grupo
    3. Ordenar de mayor a menor
    4. Retornar los primeros N elementos

    📈 VALORES POSIBLES:
    - Depende de la operación seleccionada
    - Siempre ordenados de mayor a menor

    💡 INTERPRETACIÓN:
    - Permite enfocarse en los elementos más importantes
    - Útil para priorización y toma de decisiones
    - Implementa el principio de Pareto (enfocarse en lo vital)

    📊 EJEMPLO: Top 5 Modelos por Volumen Total

    1. 7 Series: 23,786,466 unidades (9.4%)
       - Modelo más vendido
       - Líder del mercado

    2. i8: 23,423,891 unidades (9.2%)
       - Segundo lugar, muy cerca del primero

    3. X1: 23,406,060 unidades (9.2%)
       - Tercer lugar

    4. 3 Series: 23,281,303 unidades (9.2%)
    5. i3: 23,133,849 unidades (9.1%)

    🔍 ANÁLISIS:
    - Los top 5 modelos representan ~46% del total de ventas
    - Distribución muy equilibrada entre los top 5
    - Diferencia mínima entre ellos (~2.8% entre 1° y 5°)

    🎯 APLICACIONES:
    - Identificar productos estrella
    - Priorizar inventario
    - Enfocar esfuerzos de marketing
    - Análisis de mejores clientes
    - Ranking de regiones por ventas

    Args:
        datos: Lista de diccionarios
        columna_grupo: Columna para agrupar
        columna_metrica: Columna numérica para calcular métrica
        n: Número de elementos a retornar
        operacion: 'suma', 'promedio', 'max', 'min'

    Returns:
        list: Lista de tuplas (grupo, valor)
    """
    agrupado = agrupar_por_columna(datos, columna_grupo, columna_metrica, operacion)
    return sorted(agrupado.items(), key=lambda x: x[1], reverse=True)[:n]


# =============================================================================
# EJEMPLOS PRÁCTICOS
# =============================================================================

def ejemplo_estadistica_basica():
    """Ejemplos de estadística básica con datos de ventas de BMW."""
    print("\n" + "="*70)
    print("NIVEL 1: ESTADÍSTICA BÁSICA - Análisis de Ventas BMW")
    print("="*70)

    # Información del archivo
    print("\n📁 Archivo: BMW sales data (2010-2024) (1).csv")
    print("📋 Descripción: Datos de ventas de vehículos BMW desde 2010 hasta 2024,")
    print("               incluyendo modelos, regiones, precios, volumen de ventas,")
    print("               tipo de combustible, transmisión y características técnicas.")

    # Cargar datos
    ventas = leer_csv("BMW sales data (2010-2024) (1).csv")

    print(f"\n📊 Total de registros de ventas: {contar_registros(ventas):,}")
    print(f"📋 Columnas disponibles: {len(obtener_columnas(ventas))}")

    # Análisis de modelos
    print("\n--- Top 10 Modelos BMW Más Vendidos ---")
    modelos = frecuencia_valores(ventas, "Model")
    print(f"Modelos únicos: {len(modelos)}")
    for modelo, cantidad in list(sorted(modelos.items(), key=lambda x: x[1], reverse=True))[:10]:
        print(f"  • {modelo}: {cantidad:,} ventas")

    # Análisis de precios
    print("\n--- Análisis de Precios (USD) ---")
    precio_promedio = calcular_promedio(ventas, "Price_USD")
    precio_min, precio_max = encontrar_minimo_maximo(ventas, "Price_USD")
    precio_total = calcular_suma(ventas, "Price_USD")

    print(f"Precio promedio: ${precio_promedio:,.2f}")
    print(f"Precio mínimo: ${precio_min:,.2f}")
    print(f"Precio máximo: ${precio_max:,.2f}")
    print(f"Valor total de ventas: ${precio_total:,.0f}")

    # Análisis de regiones
    print("\n--- Distribución de Ventas por Región ---")
    regiones = frecuencia_valores(ventas, "Region")
    for region, cantidad in sorted(regiones.items(), key=lambda x: x[1], reverse=True):
        porcentaje = (cantidad / len(ventas)) * 100
        print(f"  • {region}: {cantidad:,} ventas ({porcentaje:.1f}%)")

    # Análisis de tipo de combustible
    print("\n--- Distribución por Tipo de Combustible ---")
    combustibles = frecuencia_valores(ventas, "Fuel_Type")
    for combustible, cantidad in sorted(combustibles.items(), key=lambda x: x[1], reverse=True):
        porcentaje = (cantidad / len(ventas)) * 100
        print(f"  • {combustible}: {cantidad:,} ventas ({porcentaje:.1f}%)")


def ejemplo_estadistica_avanzada():
    """Ejemplos de estadística avanzada con datos de ventas de BMW."""
    print("\n" + "="*70)
    print("NIVEL 2: ESTADÍSTICA AVANZADA - Análisis de Ventas BMW")
    print("="*70)

    # Información del archivo
    print("\n📁 Archivo: BMW sales data (2010-2024) (1).csv")
    print("📋 Descripción: Análisis estadístico avanzado de ventas de BMW,")
    print("               incluyendo distribuciones, percentiles y segmentaciones")
    print("               por diferentes variables (modelo, región, año, etc.).")

    # Cargar datos
    ventas = leer_csv("BMW sales data (2010-2024) (1).csv")

    # Estadísticas de precios
    print("\n--- Estadísticas Avanzadas de Precios (USD) ---")
    print("🎯 OBJETIVO: Analizar la distribución central y dispersión de los precios")
    print("📊 FUNCIONES: calcular_promedio(), calcular_mediana(), calcular_desviacion_estandar()")
    print()

    promedio = calcular_promedio(ventas, "Price_USD")
    mediana = calcular_mediana(ventas, "Price_USD")
    desv_std = calcular_desviacion_estandar(ventas, "Price_USD")

    print(f"Promedio: ${promedio:,.2f}")
    print(f"Mediana: ${mediana:,.2f}")
    print(f"Desviación estándar: ${desv_std:,.2f}")

    print("\n💡 INTERPRETACIÓN:")
    if abs(promedio - mediana) < promedio * 0.05:
        print(f"   • Promedio (${promedio:,.2f}) ≈ Mediana (${mediana:,.2f})")
        print("   → Distribución SIMÉTRICA: Los precios están balanceados")
    elif promedio > mediana:
        print(f"   • Promedio (${promedio:,.2f}) > Mediana (${mediana:,.2f})")
        print("   → Hay algunos vehículos muy caros que elevan el promedio")
    else:
        print(f"   • Promedio (${promedio:,.2f}) < Mediana (${mediana:,.2f})")
        print("   → Hay algunos vehículos muy baratos que reducen el promedio")

    coef_var = (desv_std / promedio) * 100
    print(f"   • Coeficiente de Variación: {coef_var:.2f}%")
    if coef_var < 15:
        print("   → BAJA variabilidad: Precios muy homogéneos")
    elif coef_var < 30:
        print("   → MODERADA variabilidad: Precios con cierta dispersión")
    else:
        print("   → ALTA variabilidad: Precios muy heterogéneos")
    print(f"   • Rango típico: ${promedio - desv_std:,.2f} - ${promedio + desv_std:,.2f}")
    print("     (Aproximadamente 68% de los vehículos están en este rango)")

    # Percentiles de precios
    print("\n--- Percentiles de Precios ---")
    print("🎯 OBJETIVO: Entender la distribución de precios por cuartiles")
    print("📊 FUNCIÓN: calcular_percentil()")
    print()

    p25 = calcular_percentil(ventas, "Price_USD", 25)
    p50 = calcular_percentil(ventas, "Price_USD", 50)
    p75 = calcular_percentil(ventas, "Price_USD", 75)
    p90 = calcular_percentil(ventas, "Price_USD", 90)
    p95 = calcular_percentil(ventas, "Price_USD", 95)

    print(f"P25 (25%): ${p25:,.2f}")
    print(f"P50 (50%): ${p50:,.2f}")
    print(f"P75 (75%): ${p75:,.2f}")
    print(f"P90 (90%): ${p90:,.2f}")
    print(f"P95 (95%): ${p95:,.2f}")

    print("\n💡 INTERPRETACIÓN:")
    print(f"   • El 25% de los vehículos cuestan menos de ${p25:,.2f}")
    print(f"   • El 50% de los vehículos cuestan menos de ${p50:,.2f} (mediana)")
    print(f"   • El 75% de los vehículos cuestan menos de ${p75:,.2f}")
    print(f"   • Solo el 10% cuestan más de ${p90:,.2f}")
    print(f"   • Solo el 5% cuestan más de ${p95:,.2f} (vehículos premium)")
    iqr = p75 - p25
    print(f"   • Rango Intercuartílico (IQR): ${iqr:,.2f}")
    print(f"     → El 50% central de los vehículos está entre ${p25:,.2f} y ${p75:,.2f}")

    # Agrupación por modelo - precio promedio
    print("\n--- Precio Promedio por Modelo (Top 10) ---")
    print("🎯 OBJETIVO: Comparar precios promedio entre diferentes modelos")
    print("📊 FUNCIÓN: agrupar_por_columna(columna_agrupacion='Model', operacion='promedio')")
    print()

    por_modelo = agrupar_por_columna(ventas, "Model", "Price_USD", "promedio")
    modelos_ordenados = list(sorted(por_modelo.items(), key=lambda x: x[1], reverse=True))[:10]

    for modelo, precio_prom in modelos_ordenados:
        print(f"  {modelo}: ${precio_prom:,.2f}")

    print("\n💡 INTERPRETACIÓN:")
    modelo_mas_caro = modelos_ordenados[0]
    modelo_mas_barato = modelos_ordenados[-1]
    diferencia = modelo_mas_caro[1] - modelo_mas_barato[1]
    print(f"   • Modelo más caro: {modelo_mas_caro[0]} (${modelo_mas_caro[1]:,.2f})")
    print(f"   • Modelo más económico: {modelo_mas_barato[0]} (${modelo_mas_barato[1]:,.2f})")
    print(f"   • Diferencia: ${diferencia:,.2f}")
    if diferencia < 2000:
        print("   → Los precios entre modelos son MUY SIMILARES")
    elif diferencia < 5000:
        print("   → Hay POCA diferencia de precio entre modelos")
    else:
        print("   → Hay DIFERENCIAS SIGNIFICATIVAS entre modelos")

    # Volumen de ventas por región
    print("\n--- Volumen Total de Ventas por Región ---")
    print("🎯 OBJETIVO: Identificar las regiones con mayor volumen de ventas")
    print("📊 FUNCIÓN: agrupar_por_columna(columna_agrupacion='Region', operacion='suma')")
    print()

    por_region = agrupar_por_columna(ventas, "Region", "Sales_Volume", "suma")
    regiones_ordenadas = sorted(por_region.items(), key=lambda x: x[1], reverse=True)
    total_volumen = sum(por_region.values())

    for region, volumen in regiones_ordenadas:
        porcentaje = (volumen / total_volumen) * 100
        print(f"  {region}: {volumen:,.0f} unidades ({porcentaje:.1f}%)")

    print("\n💡 INTERPRETACIÓN:")
    region_lider = regiones_ordenadas[0]
    print(f"   • Región líder: {region_lider[0]} con {region_lider[1]:,.0f} unidades")
    diferencia_max = region_lider[1] - regiones_ordenadas[-1][1]
    porcentaje_dif = (diferencia_max / region_lider[1]) * 100
    if porcentaje_dif < 5:
        print("   → Las ventas están MUY EQUILIBRADAS entre regiones")
    elif porcentaje_dif < 15:
        print("   → Las ventas están RELATIVAMENTE EQUILIBRADAS entre regiones")
    else:
        print(f"   → Hay DIFERENCIAS SIGNIFICATIVAS entre regiones ({porcentaje_dif:.1f}%)")

    # Análisis por tipo de transmisión
    print("\n--- Precio Promedio por Tipo de Transmisión ---")
    por_transmision = agrupar_por_columna(ventas, "Transmission", "Price_USD", "promedio")
    for trans, precio in sorted(por_transmision.items(), key=lambda x: x[1], reverse=True):
        print(f"  {trans}: ${precio:,.2f}")

    # Filtrado de vehículos de lujo (precio > $100,000)
    print("\n--- Vehículos de Lujo (Precio > $100,000) ---")
    lujo = filtrar_datos(ventas, "Price_USD", lambda x: x.strip() and float(x) > 100000)
    print(f"Total de vehículos de lujo: {len(lujo):,} ({(len(lujo)/len(ventas)*100):.1f}%)")
    if len(lujo) > 0:
        modelos_lujo = frecuencia_valores(lujo, "Model")
        print("Modelos de lujo más comunes:")
        for modelo, cant in list(sorted(modelos_lujo.items(), key=lambda x: x[1], reverse=True))[:5]:
            print(f"  • {modelo}: {cant:,} unidades")

    # Análisis de varianza de precios
    print("\n--- Análisis de Variabilidad de Precios ---")
    varianza = analisis_varianza(ventas, "Price_USD")
    print(f"Varianza: ${varianza:,.2f}")
    print(f"Desviación estándar: ${desv_std:,.2f}")
    coef_variacion = (desv_std / promedio) * 100
    print(f"Coeficiente de variación: {coef_variacion:.2f}%")

    # Correlación entre precio y kilometraje
    print("\n--- Correlación Precio vs Kilometraje ---")
    print("🎯 OBJETIVO: Medir si existe relación lineal entre precio y kilometraje")
    print("📊 FUNCIÓN: calcular_correlacion_numerica() - Correlación de Pearson")
    print("📈 RANGO: -1 (correlación negativa perfecta) a +1 (correlación positiva perfecta)")
    print()

    correlacion = calcular_correlacion_numerica(ventas, "Price_USD", "Mileage_KM")
    print(f"Coeficiente de correlación: {correlacion:.4f}")

    print("\n💡 INTERPRETACIÓN:")
    abs_corr = abs(correlacion)

    # Interpretación de magnitud
    if abs_corr > 0.7:
        fuerza = "FUERTE"
    elif abs_corr > 0.3:
        fuerza = "MODERADA"
    else:
        fuerza = "DÉBIL o NULA"

    # Interpretación de dirección
    if correlacion > 0.3:
        print(f"   • Correlación positiva {fuerza.lower()}")
        print("   → Cuando el kilometraje aumenta, el precio también tiende a aumentar")
    elif correlacion < -0.3:
        print(f"   • Correlación negativa {fuerza.lower()}")
        print("   → Cuando el kilometraje aumenta, el precio tiende a disminuir")
    else:
        print(f"   • Correlación {fuerza}")
        print("   → NO hay relación lineal significativa entre precio y kilometraje")
        print("   → El kilometraje NO es un factor determinante del precio")
        print("   → Otros factores (modelo, año, región) pueden ser más importantes")

    print(f"\n   ⚠️  IMPORTANTE: Correlación NO implica causalidad")

    # Análisis por rangos de precio
    print("\n--- Distribución por Rangos de Precio ---")
    print("🎯 OBJETIVO: Clasificar vehículos en segmentos de precio personalizados")
    print("📊 FUNCIÓN: analisis_por_rango()")
    print()

    rangos_precio = [
        (30000, 50000, "$30K-$50K"),
        (50000, 70000, "$50K-$70K"),
        (70000, 90000, "$70K-$90K"),
        (90000, 110000, "$90K-$110K"),
        (110000, 130000, "$110K+")
    ]
    distribucion = analisis_por_rango(ventas, "Price_USD", rangos_precio)

    etiquetas_segmento = {
        "$30K-$50K": "Económico",
        "$50K-$70K": "Medio-Bajo",
        "$70K-$90K": "Medio-Alto",
        "$90K-$110K": "Premium",
        "$110K+": "Lujo"
    }

    for rango, cantidad in distribucion.items():
        porcentaje = (cantidad / len(ventas)) * 100
        etiqueta = etiquetas_segmento.get(rango, "")
        print(f"  {rango} ({etiqueta}): {cantidad:,} vehículos ({porcentaje:.1f}%)")

    print("\n💡 INTERPRETACIÓN:")
    max_rango = max(distribucion.items(), key=lambda x: x[1])
    min_rango = min(distribucion.items(), key=lambda x: x[1])
    print(f"   • Segmento más popular: {max_rango[0]} con {max_rango[1]:,} vehículos")
    print(f"   • Segmento menos popular: {min_rango[0]} con {min_rango[1]:,} vehículos")

    # Calcular si la distribución es equilibrada
    valores = list(distribucion.values())
    promedio_dist = sum(valores) / len(valores)
    desv_dist = (sum((v - promedio_dist) ** 2 for v in valores) / len(valores)) ** 0.5
    cv_dist = (desv_dist / promedio_dist) * 100

    if cv_dist < 10:
        print("   → Distribución MUY EQUILIBRADA entre segmentos")
    elif cv_dist < 25:
        print("   → Distribución RELATIVAMENTE EQUILIBRADA entre segmentos")
    else:
        print("   → Distribución DESBALANCEADA: Algunos segmentos dominan el mercado")


def ejemplo_ciencia_datos():
    """Ejemplos de ciencia de datos con datos de ventas de BMW."""
    print("\n" + "="*70)
    print("NIVEL 3: CIENCIA DE DATOS - Análisis Avanzado BMW")
    print("="*70)

    # Información de los archivos
    print("\n📁 Archivo: BMW sales data (2010-2024) (1).csv")
    print("📋 Descripción: Análisis avanzado de ciencia de datos aplicado a ventas BMW,")
    print("               incluyendo análisis temporal, detección de outliers, análisis ABC,")
    print("               segmentación de mercado y correlaciones entre variables.")

    # Cargar datos
    ventas = leer_csv("BMW sales data (2010-2024) (1).csv")

    # Análisis temporal por año
    print("\n--- Análisis Temporal de Ventas por Año ---")
    tendencia = analisis_temporal(ventas, "Year", "Sales_Volume")
    print("Evolución anual del volumen de ventas:")
    for año, stats in list(sorted(tendencia.items()))[:10]:  # Primeros 10 años
        print(f"  {año}: {stats['total']:,.0f} unidades (promedio: {stats['promedio']:,.0f})")

    # Tasa de crecimiento anual
    print("\n--- Tasa de Crecimiento Anual de Ventas ---")
    crecimiento = calcular_tasa_crecimiento(ventas, "Year", "Sales_Volume")
    for periodo, valor, tasa in crecimiento[:10]:
        signo = "+" if tasa >= 0 else ""
        print(f"  {periodo}: {valor:,.0f} unidades ({signo}{tasa:.1f}%)")

    # Detección de outliers en precios
    print("\n--- Detección de Valores Atípicos en Precios ---")
    print("🎯 OBJETIVO: Identificar precios anormalmente altos o bajos")
    print("📊 FUNCIÓN: detectar_outliers() - Método IQR (Rango Intercuartílico)")
    print("📈 ALGORITMO: Outliers = valores fuera de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]")
    print()

    outliers_info = detectar_outliers(ventas, "Price_USD")
    print(f"Outliers detectados: {outliers_info['cantidad']:,}")
    print(f"Rango normal: ${outliers_info['limite_inferior']:,.2f} - ${outliers_info['limite_superior']:,.2f}")
    if outliers_info['outliers']:
        print(f"Primeros 5 valores atípicos: {[f'${x:,.2f}' for x in outliers_info['outliers'][:5]]}")

    print("\n💡 INTERPRETACIÓN:")
    porcentaje_outliers = (outliers_info['cantidad'] / len(ventas)) * 100
    print(f"   • Porcentaje de outliers: {porcentaje_outliers:.2f}%")

    if outliers_info['cantidad'] == 0:
        print("   → NO hay valores atípicos detectados")
        print("   → Todos los precios están dentro del rango esperado")
        print("   → Los datos son muy CONSISTENTES y HOMOGÉNEOS")
    elif porcentaje_outliers < 1:
        print("   → MUY POCOS outliers (menos del 1%)")
        print("   → Los datos son generalmente consistentes")
        print("   → Los outliers pueden ser casos especiales legítimos")
    elif porcentaje_outliers < 5:
        print("   → POCOS outliers (menos del 5%)")
        print("   → Cantidad normal de valores extremos")
    else:
        print("   → MUCHOS outliers (más del 5%)")
        print("   → Revisar la calidad de los datos")
        print("   → Puede indicar múltiples segmentos de mercado")

    # Análisis ABC de modelos por volumen de ventas
    print("\n--- Análisis ABC de Modelos por Volumen de Ventas ---")
    print("🎯 OBJETIVO: Clasificar modelos según el Principio de Pareto (80/20)")
    print("📊 FUNCIÓN: analisis_abc()")
    print("📈 CATEGORÍAS:")
    print("   • A: Modelos que generan el 80% de las ventas (los 'pocos vitales')")
    print("   • B: Modelos que generan el 15% de las ventas (importancia media)")
    print("   • C: Modelos que generan el 5% de las ventas (los 'muchos triviales')")
    print()

    abc = analisis_abc(ventas, "Model", "Sales_Volume")

    # Contar por categoría
    categorias_abc = {'A': 0, 'B': 0, 'C': 0}
    for _, info in abc.items():
        categorias_abc[info['categoria']] += 1

    print(f"Categoría A (80% de las ventas): {categorias_abc['A']} modelos")
    print(f"Categoría B (15% de las ventas): {categorias_abc['B']} modelos")
    print(f"Categoría C (5% de las ventas): {categorias_abc['C']} modelos")

    # Mostrar top 5 modelos categoría A
    print("\nTop 5 modelos categoría A (más vendidos):")
    items_a = [(k, v) for k, v in abc.items() if v['categoria'] == 'A']
    for modelo, info in sorted(items_a, key=lambda x: x[1]['valor'], reverse=True)[:5]:
        print(f"  • {modelo}: {info['valor']:,.0f} unidades ({info['porcentaje_individual']:.1f}%)")

    print("\n💡 INTERPRETACIÓN:")
    total_modelos = len(abc)
    porcentaje_a = (categorias_abc['A'] / total_modelos) * 100
    print(f"   • Solo el {porcentaje_a:.1f}% de los modelos ({categorias_abc['A']} de {total_modelos})")
    print("     generan el 80% de las ventas totales")
    print("   → Estos son los modelos ESTRELLA que impulsan el negocio")
    print("   → Requieren MÁXIMA atención en:")
    print("     - Gestión de inventario (mantener stock suficiente)")
    print("     - Marketing (enfocar campañas publicitarias)")
    print("     - Producción (optimizar procesos)")
    print("     - Servicio al cliente (capacitación especializada)")
    print(f"\n   • Los modelos categoría C ({categorias_abc['C']} modelos)")
    print("     generan solo el 5% de las ventas")
    print("   → Considerar descontinuar o reducir inversión en estos modelos")

    # Segmentación de precios
    print("\n--- Segmentación de Vehículos por Precio ---")
    print("🎯 OBJETIVO: Dividir vehículos en 3 segmentos de precio automáticos")
    print("📊 FUNCIÓN: crear_segmentos(num_segmentos=3)")
    print("📈 ALGORITMO: Divide el rango de precios en 3 partes iguales")
    print()

    segmentos = crear_segmentos(ventas, "Price_USD", 3)
    conteo_segmentos = Counter(seg for _, seg in segmentos)

    etiquetas = {0: "Económico", 1: "Medio", 2: "Premium"}
    for seg in sorted(conteo_segmentos.keys()):
        porcentaje = (conteo_segmentos[seg] / len(ventas)) * 100
        print(f"  Segmento {etiquetas[seg]}: {conteo_segmentos[seg]:,} vehículos ({porcentaje:.1f}%)")

    print("\n💡 INTERPRETACIÓN:")
    # Calcular si la distribución es equilibrada
    valores_seg = list(conteo_segmentos.values())
    max_seg = max(valores_seg)
    min_seg = min(valores_seg)
    diferencia_seg = ((max_seg - min_seg) / max_seg) * 100

    if diferencia_seg < 5:
        print("   → Distribución MUY EQUILIBRADA entre segmentos")
        print("   → El mercado está balanceado en todos los rangos de precio")
    elif diferencia_seg < 15:
        print("   → Distribución RELATIVAMENTE EQUILIBRADA")
        print("   → Hay demanda en todos los segmentos de precio")
    else:
        print("   → Distribución DESBALANCEADA")
        print("   → Algunos segmentos dominan el mercado")

    print("\n   💼 APLICACIÓN PRÁCTICA:")
    print("   • Usar esta segmentación para:")
    print("     - Estrategias de marketing diferenciadas por segmento")
    print("     - Gestión de inventario por categoría de precio")
    print("     - Análisis de rentabilidad por segmento")
    print("     - Identificación de oportunidades de mercado")

    # Correlación categórica
    print("\n--- Relación entre Tipo de Combustible y Clasificación de Ventas ---")
    print("🎯 OBJETIVO: Analizar si el tipo de combustible influye en la clasificación de ventas")
    print("📊 FUNCIÓN: correlacion_categorica() - Tabla de contingencia")
    print()

    correlacion = correlacion_categorica(ventas, "Fuel_Type", "Sales_Classification")

    # Calcular totales por combustible
    totales_combustible = {}
    for combustible, clasificaciones in correlacion.items():
        totales_combustible[combustible] = sum(clasificaciones.values())

    for combustible, clasificaciones in sorted(correlacion.items()):
        total_comb = totales_combustible[combustible]
        print(f"  {combustible} (Total: {total_comb:,}):")
        for clasificacion, cantidad in sorted(clasificaciones.items(), key=lambda x: x[1], reverse=True):
            porcentaje = (cantidad / total_comb) * 100
            print(f"    - {clasificacion}: {cantidad:,} ventas ({porcentaje:.1f}%)")

    print("\n💡 INTERPRETACIÓN:")
    # Analizar si hay patrones
    print("   • Analizando la distribución de clasificaciones por tipo de combustible:")

    # Verificar si las distribuciones son similares
    distribuciones_similares = True
    for combustible, clasificaciones in correlacion.items():
        total = totales_combustible[combustible]
        for _, cant in clasificaciones.items():
            porcentaje = (cant / total) * 100
            # Si algún porcentaje se desvía mucho de 50%, hay diferencia
            if abs(porcentaje - 50) > 15:  # Más de 15% de desviación
                distribuciones_similares = False

    if distribuciones_similares:
        print("   → Las distribuciones son SIMILARES entre tipos de combustible")
        print("   → NO hay relación fuerte entre tipo de combustible y clasificación")
        print("   → El tipo de combustible NO determina si las ventas serán altas o bajas")
    else:
        print("   → Las distribuciones son DIFERENTES entre tipos de combustible")
        print("   → SÍ hay relación entre tipo de combustible y clasificación")
        print("   → Algunos tipos de combustible tienen mejor desempeño en ventas")

    # Análisis adicional: Relación entre Región y Modelo
    print("\n--- Top 3 Modelos Más Vendidos por Región ---")
    for region in sorted(list(frecuencia_valores(ventas, "Region").keys())):
        ventas_region = [v for v in ventas if v.get("Region") == region]
        if ventas_region:
            modelos_region = frecuencia_valores(ventas_region, "Model")
            print(f"  {region}:")
            for modelo, cant in list(sorted(modelos_region.items(), key=lambda x: x[1], reverse=True))[:3]:
                print(f"    - {modelo}: {cant:,} ventas")

    # Análisis de tendencias: Modelos con mayor crecimiento
    print("\n--- Análisis de Tendencias: Crecimiento por Modelo ---")
    print("🎯 OBJETIVO: Identificar modelos con mayor crecimiento en 14 años")
    print("📊 FUNCIÓN: calcular_tasa_crecimiento() + agrupar_por_columna()")
    print("📈 FÓRMULA: Tasa = ((Valor_2024 - Valor_2010) / Valor_2010) × 100%")
    print("\n(Comparación 2010 vs 2024)")

    ventas_2010 = [v for v in ventas if v.get("Year") == "2010"]
    ventas_2024 = [v for v in ventas if v.get("Year") == "2024"]

    if ventas_2010 and ventas_2024:
        modelos_2010 = agrupar_por_columna(ventas_2010, "Model", "Sales_Volume", "suma")
        modelos_2024 = agrupar_por_columna(ventas_2024, "Model", "Sales_Volume", "suma")

        crecimiento_modelos = []
        for modelo in modelos_2010.keys():
            if modelo in modelos_2024:
                vol_2010 = modelos_2010[modelo]
                vol_2024 = modelos_2024[modelo]
                if vol_2010 > 0:
                    tasa_crecimiento = ((vol_2024 - vol_2010) / vol_2010) * 100
                    crecimiento_modelos.append((modelo, vol_2010, vol_2024, tasa_crecimiento))

        print("\nTop 5 modelos con mayor crecimiento:")
        top_5_crecimiento = sorted(crecimiento_modelos, key=lambda x: x[3], reverse=True)[:5]
        for modelo, vol_2010, vol_2024, tasa in top_5_crecimiento:
            signo = "+" if tasa >= 0 else ""
            print(f"  {modelo}: {signo}{tasa:.1f}%")
            print(f"    2010: {vol_2010:,.0f} → 2024: {vol_2024:,.0f}")

        print("\n💡 INTERPRETACIÓN:")
        mejor_modelo = top_5_crecimiento[0]
        print(f"   • Modelo con MAYOR crecimiento: {mejor_modelo[0]} ({mejor_modelo[3]:+.1f}%)")
        if mejor_modelo[3] > 20:
            print("   → Crecimiento FUERTE: Modelo en expansión")
        elif mejor_modelo[3] > 5:
            print("   → Crecimiento MODERADO: Modelo estable con tendencia positiva")
        else:
            print("   → Crecimiento BAJO: Mercado maduro")

        promedio_crecimiento = sum(x[3] for x in crecimiento_modelos) / len(crecimiento_modelos)
        print(f"\n   • Tasa de crecimiento promedio de todos los modelos: {promedio_crecimiento:+.1f}%")
        if promedio_crecimiento > 10:
            print("   → El mercado BMW está en EXPANSIÓN general")
        elif promedio_crecimiento > 0:
            print("   → El mercado BMW tiene crecimiento MODERADO")
        else:
            print("   → El mercado BMW está en CONTRACCIÓN")

    # Top modelos por volumen total usando la nueva función
    print("\n--- Top 5 Modelos por Volumen Total de Ventas ---")
    print("🎯 OBJETIVO: Identificar los 5 modelos más vendidos de todos los tiempos")
    print("📊 FUNCIÓN: top_n_por_metrica(n=5, operacion='suma')")
    print()

    top_modelos = top_n_por_metrica(ventas, "Model", "Sales_Volume", n=5, operacion="suma")
    total_general = sum(volumen for _, volumen in top_modelos)

    for i, (modelo, volumen) in enumerate(top_modelos, 1):
        porcentaje = (volumen / sum(v for _, v in top_modelos)) * 100
        print(f"  {i}. {modelo}: {volumen:,.0f} unidades ({porcentaje:.1f}%)")

    print("\n💡 INTERPRETACIÓN:")
    print(f"   • Los Top 5 modelos representan {total_general:,.0f} unidades vendidas")

    # Verificar si la distribución es equilibrada
    volumenes = [v for _, v in top_modelos]
    max_vol = max(volumenes)
    min_vol = min(volumenes)
    diferencia_pct = ((max_vol - min_vol) / max_vol) * 100

    if diferencia_pct < 5:
        print("   → Distribución MUY EQUILIBRADA entre los top 5")
        print("   → No hay un modelo claramente dominante")
    elif diferencia_pct < 15:
        print("   → Distribución RELATIVAMENTE EQUILIBRADA")
        print("   → Varios modelos compiten por el liderazgo")
    else:
        print("   → Distribución DESBALANCEADA")
        print(f"   → El modelo líder ({top_modelos[0][0]}) domina claramente")

    print("\n   💼 APLICACIÓN PRÁCTICA:")
    print("   • Enfocar recursos de marketing en estos 5 modelos")
    print("   • Garantizar disponibilidad de inventario")
    print("   • Capacitar al equipo de ventas en estos modelos prioritarios")


def ejemplo_cruce_datos():
    """Ejemplo de análisis multidimensional con datos de BMW."""
    print("\n" + "="*70)
    print("BONUS: ANÁLISIS MULTIDIMENSIONAL - Insights Avanzados BMW")
    print("="*70)

    # Información del archivo
    print("\n📁 Archivo: BMW sales data (2010-2024) (1).csv")
    print("📋 Descripción: Análisis cruzado de múltiples dimensiones para obtener")
    print("               insights profundos sobre el comportamiento de ventas BMW,")
    print("               combinando variables como modelo, región, año, combustible, etc.")

    # Cargar datos
    print("\nCargando datos...")
    ventas = leer_csv("BMW sales data (2010-2024) (1).csv")
    print(f"Total de registros: {len(ventas):,}")

    # Análisis 1: Precio promedio por modelo y región
    print("\n--- Precio Promedio por Modelo en Cada Región (Top 5 Modelos) ---")
    modelos_top = list(frecuencia_valores(ventas, "Model").keys())[:5]
    regiones = sorted(list(frecuencia_valores(ventas, "Region").keys()))

    for modelo in modelos_top:
        print(f"\n  {modelo}:")
        ventas_modelo = [v for v in ventas if v.get("Model") == modelo]
        for region in regiones:
            ventas_modelo_region = [v for v in ventas_modelo if v.get("Region") == region]
            if ventas_modelo_region:
                precio_prom = calcular_promedio(ventas_modelo_region, "Price_USD")
                cantidad = len(ventas_modelo_region)
                print(f"    {region}: ${precio_prom:,.2f} ({cantidad:,} ventas)")

    # Análisis 2: Evolución de ventas por tipo de combustible a lo largo de los años
    print("\n--- Evolución de Ventas por Tipo de Combustible (2010-2024) ---")
    combustibles = sorted(list(frecuencia_valores(ventas, "Fuel_Type").keys()))
    años_muestra = ['2010', '2015', '2020', '2024']

    for año in años_muestra:
        print(f"\n  Año {año}:")
        ventas_año = [v for v in ventas if v.get("Year") == año]
        if ventas_año:
            for combustible in combustibles:
                ventas_comb = [v for v in ventas_año if v.get("Fuel_Type") == combustible]
                if ventas_comb:
                    volumen = calcular_suma(ventas_comb, "Sales_Volume")
                    porcentaje = (len(ventas_comb) / len(ventas_año)) * 100
                    print(f"    {combustible}: {volumen:,.0f} unidades ({porcentaje:.1f}%)")

    # Análisis 3: Modelos más rentables por región (precio * volumen)
    print("\n--- Top 3 Modelos Más Rentables por Región ---")
    for region in regiones:
        print(f"\n  {region}:")
        ventas_region = [v for v in ventas if v.get("Region") == region]

        # Calcular rentabilidad por modelo
        rentabilidad_modelo = defaultdict(float)
        for venta in ventas_region:
            modelo = venta.get("Model", "")
            try:
                precio = float(venta.get("Price_USD", 0))
                volumen = float(venta.get("Sales_Volume", 0))
                rentabilidad_modelo[modelo] += precio * volumen
            except ValueError:
                continue

        # Mostrar top 3
        for modelo, rentabilidad in list(sorted(rentabilidad_modelo.items(), key=lambda x: x[1], reverse=True))[:3]:
            print(f"    {modelo}: ${rentabilidad:,.0f}")

    # Análisis 4: Comparación de transmisión por modelo
    print("\n--- Preferencia de Transmisión por Modelo (Top 5 Modelos) ---")
    for modelo in modelos_top:
        print(f"\n  {modelo}:")
        ventas_modelo = [v for v in ventas if v.get("Model") == modelo]
        transmisiones = frecuencia_valores(ventas_modelo, "Transmission")
        total_modelo = len(ventas_modelo)
        for trans, cant in sorted(transmisiones.items(), key=lambda x: x[1], reverse=True):
            porcentaje = (cant / total_modelo) * 100
            print(f"    {trans}: {cant:,} ventas ({porcentaje:.1f}%)")

    # Análisis 5: Vehículos con mejor relación precio-kilometraje
    print("\n--- Top 10 Vehículos con Mejor Relación Precio/Kilometraje ---")
    print("(Menor precio por kilómetro recorrido)")
    relacion_precio_km = []
    for venta in ventas:
        try:
            precio = float(venta.get("Price_USD", 0))
            kilometraje = float(venta.get("Mileage_KM", 0))
            if kilometraje > 0:
                relacion = precio / kilometraje
                relacion_precio_km.append({
                    'modelo': venta.get("Model", ""),
                    'año': venta.get("Year", ""),
                    'precio': precio,
                    'km': kilometraje,
                    'relacion': relacion
                })
        except ValueError:
            continue

    for item in sorted(relacion_precio_km, key=lambda x: x['relacion'])[:10]:
        print(f"  {item['modelo']} ({item['año']}): ${item['relacion']:.2f}/km")
        print(f"    Precio: ${item['precio']:,.0f} | Kilometraje: {item['km']:,.0f} km")


# =============================================================================
# MENÚ INTERACTIVO
# =============================================================================

def mostrar_menu():
    """Muestra el menú principal."""
    print("\n" + "="*70)
    print("🎓 ACTIVIDAD 8: ANÁLISIS DE DATOS CON CSV")
    print("="*70)
    print("\nSelecciona un nivel de análisis:")
    print("\n  1️⃣  Nivel 1: Estadística Básica")
    print("      (Conteos, promedios, frecuencias)")
    print("\n  2️⃣  Nivel 2: Estadística Avanzada")
    print("      (Mediana, desviación estándar, percentiles, agrupaciones)")
    print("\n  3️⃣  Nivel 3: Ciencia de Datos")
    print("      (Análisis temporal, outliers, ABC, segmentación)")
    print("\n  4️⃣  Bonus: Cruce de Datos")
    print("      (Integración de múltiples datasets)")
    print("\n  5️⃣  Ejecutar Todos los Ejemplos")
    print("\n  0️⃣  Salir")
    print("\n" + "="*70)


def menu_interactivo():
    """Menú interactivo para explorar los diferentes niveles."""
    while True:
        mostrar_menu()

        try:
            opcion = input("\n👉 Ingresa tu opción: ").strip()

            if opcion == "0":
                print("\n¡Hasta luego! 👋")
                break
            elif opcion == "1":
                ejemplo_estadistica_basica()
            elif opcion == "2":
                ejemplo_estadistica_avanzada()
            elif opcion == "3":
                ejemplo_ciencia_datos()
            elif opcion == "4":
                ejemplo_cruce_datos()
            elif opcion == "5":
                ejemplo_estadistica_basica()
                input("\n⏸️  Presiona Enter para continuar...")
                ejemplo_estadistica_avanzada()
                input("\n⏸️  Presiona Enter para continuar...")
                ejemplo_ciencia_datos()
                input("\n⏸️  Presiona Enter para continuar...")
                ejemplo_cruce_datos()
            else:
                print("\n❌ Opción no válida. Por favor, intenta de nuevo.")

            if opcion in ["1", "2", "3", "4"]:
                input("\n⏸️  Presiona Enter para volver al menú...")

        except KeyboardInterrupt:
            print("\n\n¡Hasta luego! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("\n⏸️  Presiona Enter para continuar...")


# =============================================================================
# EJERCICIOS PROPUESTOS
# =============================================================================

def mostrar_ejercicios():
    """Muestra ejercicios propuestos para practicar con datos de BMW."""
    print("\n" + "="*70)
    print("📝 EJERCICIOS PROPUESTOS - ANÁLISIS DE VENTAS BMW")
    print("="*70)

    print("\n📁 Archivo de datos: BMW sales data (2010-2024) (1).csv")
    print("📊 Columnas disponibles: Model, Year, Region, Color, Fuel_Type,")
    print("   Transmission, Engine_Size_L, Mileage_KM, Price_USD,")
    print("   Sales_Volume, Sales_Classification")

    ejercicios = [
        {
            "nivel": "Básico",
            "emoji": "🟢",
            "ejercicios": [
                "1. Encuentra el color de vehículo más popular en las ventas",
                "2. Calcula el precio promedio de los vehículos eléctricos (Electric)",
                "3. Cuenta cuántos vehículos tienen transmisión automática vs manual",
                "4. Encuentra el tamaño de motor (Engine_Size_L) más común",
                "5. Calcula el kilometraje promedio de todos los vehículos vendidos"
            ]
        },
        {
            "nivel": "Intermedio",
            "emoji": "🟡",
            "ejercicios": [
                "1. Calcula la mediana de precios para cada tipo de combustible",
                "2. Agrupa por región y calcula el volumen total de ventas de cada una",
                "3. Filtra los vehículos con kilometraje mayor a 150,000 km",
                "4. Calcula el percentil 90 de los precios por modelo",
                "5. Encuentra la correlación entre tamaño de motor y precio",
                "6. Crea rangos de kilometraje (0-50K, 50K-100K, etc.) y cuenta vehículos"
            ]
        },
        {
            "nivel": "Avanzado",
            "emoji": "🔴",
            "ejercicios": [
                "1. Realiza un análisis ABC de los modelos por valor total de ventas",
                "2. Detecta outliers en los precios usando el método IQR",
                "3. Analiza la tendencia de ventas por año (2010-2024) para vehículos híbridos",
                "4. Calcula la tasa de crecimiento anual de ventas por región",
                "5. Crea segmentos de precio (económico, medio, premium) y analiza preferencias",
                "6. Compara la evolución de ventas de vehículos eléctricos vs gasolina",
                "7. Identifica qué modelo tiene el mejor precio promedio por región",
                "8. Analiza la relación entre clasificación de ventas (High/Low) y tipo de combustible"
            ]
        },
        {
            "nivel": "Desafío Extra",
            "emoji": "🏆",
            "ejercicios": [
                "1. Crea un análisis completo de rentabilidad por modelo (precio × volumen)",
                "2. Predice qué tipo de combustible será más popular en los próximos años",
                "3. Identifica patrones de preferencia de color por región",
                "4. Analiza la depreciación: relación entre año, kilometraje y precio",
                "5. Crea un dashboard de métricas clave para cada región"
            ]
        }
    ]

    for grupo in ejercicios:
        print(f"\n{grupo['emoji']} {grupo['nivel']}:")
        for ejercicio in grupo['ejercicios']:
            print(f"   {ejercicio}")

    print("\n" + "="*70)
    print("💡 Tips:")
    print("   • Usa las funciones ya creadas como base para resolver los ejercicios")
    print("   • Combina múltiples funciones para análisis más complejos")
    print("   • Experimenta con diferentes columnas y métricas")
    print("   • Visualiza los resultados de forma clara y organizada")
    print("="*70)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal del programa."""
    print("\n" + "🌟"*35)
    print("   BIENVENIDO AL ANÁLISIS DE DATOS CON CSV")
    print("🌟"*35)

    print("\nEste programa te enseñará a analizar datos CSV en tres niveles:")
    print("  📊 Estadística Básica")
    print("  📈 Estadística Avanzada")
    print("  🔬 Ciencia de Datos")

    print("\n¿Qué te gustaría hacer?")
    print("  1. Ver ejemplos interactivos")
    print("  2. Ver ejercicios propuestos")
    print("  3. Ambos")

    try:
        opcion = input("\n👉 Ingresa tu opción (1-3): ").strip()

        if opcion == "1":
            menu_interactivo()
        elif opcion == "2":
            mostrar_ejercicios()
        elif opcion == "3":
            menu_interactivo()
            mostrar_ejercicios()
        else:
            print("\n❌ Opción no válida")

    except KeyboardInterrupt:
        print("\n\n¡Hasta luego! 👋")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

