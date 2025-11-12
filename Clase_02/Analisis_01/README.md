# 📊 Análisis 01: Análisis Manual de Datos CSV

## 📚 Descripción General

Este directorio contiene un sistema completo de análisis de datos implementado **manualmente** usando solo Python estándar, sin dependencias de librerías externas de alto nivel.

**Propósito educativo:** Entender cómo funcionan los algoritmos estadísticos y de ciencia de datos internamente, implementándolos desde cero.

---

## 🎯 Objetivos de Aprendizaje

- ✅ Entender cómo funcionan los algoritmos estadísticos internamente
- ✅ Practicar programación con estructuras de datos básicas
- ✅ Desarrollar lógica de programación
- ✅ Aprender a leer y procesar archivos CSV manualmente
- ✅ Implementar funciones de análisis de datos desde cero
- ✅ Comprender los fundamentos antes de usar librerías profesionales

---

## 📁 Estructura del Proyecto

```
Clase_02/Analisis_01/
│
├── 📂 sample/
│   └── BMW sales data (2010-2024) (1).csv  # Dataset de 50,000 registros
│
├── 🐍 main.py                              # ⭐ Sistema completo (2,157 líneas)
│
└── 📄 README.md                            # Este archivo
```

**Total:** 2,157 líneas de código | 50,000 registros | 11 columnas | 15 años de datos

---

## 📋 Características del Sistema

- ✅ **2,157 líneas** de código Python puro
- ✅ **Sin dependencias externas** (solo módulos estándar: `csv`, `os`, `datetime`, `collections`)
- ✅ **25+ funciones** implementadas desde cero
- ✅ **3 niveles** de análisis progresivo
- ✅ **50,000 registros** de ventas de BMW (2010-2024)
- ✅ **Educativo:** Código comentado y explicado

---

## 🚀 Inicio Rápido

```bash
cd Clase_02/Analisis_01
python main.py
```

El script ejecutará automáticamente todos los niveles de análisis y mostrará los resultados en consola.

---

## 📊 Niveles de Análisis Implementados

### 📊 Nivel 1: Estadística Básica
Funciones fundamentales para análisis exploratorio de datos:

- **Lectura de archivos CSV** - Manejo de diferentes encodings
- **Conteo de registros** - Total de filas en el dataset
- **Cálculo de promedios** - Media aritmética
- **Sumas totales** - Agregación de valores numéricos
- **Mínimos y máximos** - Valores extremos
- **Frecuencias** - Distribución de valores categóricos
- **Valores únicos** - Identificación de categorías distintas

### 📈 Nivel 2: Estadística Avanzada
Análisis estadístico más profundo:

- **Mediana** - Valor central de la distribución
- **Moda** - Valor más frecuente
- **Desviación estándar** - Medida de dispersión
- **Varianza** - Dispersión cuadrática
- **Percentiles** - Cuartiles y percentiles personalizados
- **Coeficiente de variación** - Dispersión relativa
- **Agrupaciones** - Group by manual
- **Filtrado avanzado** - Condiciones complejas
- **Correlaciones** - Relación entre variables numéricas

### 🔬 Nivel 3: Ciencia de Datos
Técnicas avanzadas de análisis:

- **Análisis temporal** - Tendencias a lo largo del tiempo
- **Detección de outliers** - Método IQR (Rango Intercuartílico)
- **Análisis ABC/Pareto** - Clasificación por importancia (80/20)
- **Segmentación de datos** - Creación de grupos
- **Tasas de crecimiento** - Crecimiento año a año
- **Correlaciones categóricas** - Chi-cuadrado manual
- **Análisis de tendencias** - Regresión lineal simple
- **Rankings** - Top N por diferentes criterios
- **Análisis multidimensional** - Cruces de múltiples variables

## 💻 Funciones Principales Implementadas

### Estadísticas Básicas (Nivel 1)
```python
leer_csv()                    # Lectura de archivos CSV con manejo de encodings
contar_registros()            # Conteo total de registros
calcular_promedio()           # Media aritmética
calcular_suma()               # Suma total de valores
encontrar_minimo()            # Valor mínimo
encontrar_maximo()            # Valor máximo
frecuencia_valores()          # Distribución de frecuencias
valores_unicos()              # Valores distintos en una columna
```

### Estadísticas Avanzadas (Nivel 2)
```python
calcular_mediana()            # Mediana (valor central)
calcular_moda()               # Moda (valor más frecuente)
calcular_desviacion_estandar() # Desviación estándar
calcular_varianza()           # Varianza
calcular_percentil()          # Percentiles y cuartiles
coeficiente_variacion()       # CV% (dispersión relativa)
agrupar_por()                 # Agrupación manual (group by)
filtrar_datos()               # Filtrado con condiciones
calcular_correlacion_numerica() # Correlación de Pearson
```

### Ciencia de Datos (Nivel 3)
```python
analisis_temporal()           # Tendencias a lo largo del tiempo
detectar_outliers()           # Detección de valores atípicos (IQR)
analisis_abc()                # Clasificación ABC/Pareto (80/20)
segmentar_datos()             # Segmentación en grupos
calcular_tasa_crecimiento()   # Crecimiento año a año
correlacion_categorica()      # Chi-cuadrado manual
analisis_tendencia()          # Regresión lineal simple
ranking_top_n()               # Top N por criterio
analisis_multidimensional()   # Cruces de múltiples variables
```

---

## 📊 Dataset: Ventas de BMW (2010-2024)

### Información del Dataset
- **Registros**: 50,000 ventas
- **Período**: 2010-2024 (15 años)
- **Columnas**: 11 variables

### Columnas Disponibles

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `Model` | Categórica | Modelo del vehículo (11 modelos) |
| `Year` | Numérica | Año de venta (2010-2024) |
| `Region` | Categórica | Región de venta (6 regiones) |
| `Color` | Categórica | Color del vehículo |
| `Fuel_Type` | Categórica | Tipo de combustible (4 tipos) |
| `Transmission` | Categórica | Tipo de transmisión (2 tipos) |
| `Engine_Size_L` | Numérica | Tamaño del motor en litros |
| `Mileage_KM` | Numérica | Kilometraje del vehículo |
| `Price_USD` | Numérica | Precio en dólares |
| `Sales_Volume` | Numérica | Volumen de ventas |
| `Sales_Classification` | Categórica | Clasificación (High/Low) |

### Modelos Incluidos
- 3 Series, 5 Series, 7 Series
- X1, X3, X5, X6
- M3, M5
- i3, i8

### Regiones
- Asia
- Europe
- North America
- Middle East
- Africa
- South America

### Tipos de Combustible
- Petrol (Gasolina)
- Diesel
- Electric (Eléctrico)
- Hybrid (Híbrido)

---

## 🎯 Ejercicios Propuestos

### 🟢 Nivel Básico
1. Encuentra el color de vehículo más popular
2. Calcula el precio promedio de vehículos eléctricos
3. Cuenta transmisiones automáticas vs manuales
4. Encuentra el tamaño de motor más común
5. Calcula el kilometraje promedio por región

### 🟡 Nivel Intermedio
1. Calcula la mediana de precios por tipo de combustible
2. Agrupa por región y calcula volumen total de ventas
3. Filtra vehículos con más de 150,000 km
4. Calcula el percentil 90 de precios por modelo
5. Encuentra la correlación entre tamaño de motor y precio
6. Crea rangos de kilometraje (bajo, medio, alto)

### 🔴 Nivel Avanzado
1. Realiza análisis ABC por valor total de ventas
2. Detecta outliers en precios usando método IQR
3. Analiza la tendencia de vehículos híbridos (2010-2024)
4. Calcula la tasa de crecimiento anual por región
5. Crea segmentos de precio (económico, medio, premium)
6. Compara la evolución de eléctricos vs gasolina

---

## 💡 Ventajas del Enfoque Manual

### ✅ Aprendizaje Profundo
- **Entiendes los algoritmos internamente** - Sabes exactamente qué hace cada línea
- **Desarrollas lógica de programación** - Mejoras tus habilidades de resolución de problemas
- **Base sólida** - Preparación para entender librerías profesionales

### ✅ Sin Dependencias
- **No requiere instalación** - Solo Python estándar
- **Portable** - Funciona en cualquier entorno Python
- **Simple** - Sin conflictos de versiones

### ✅ Educativo
- **Código transparente** - Cada paso es visible
- **Fácil de modificar** - Puedes adaptar las funciones a tus necesidades
- **Ideal para enseñanza** - Perfecto para cursos y tutoriales

---

## 🔄 Comparación con Análisis Profesional

Si quieres ver cómo se hace el mismo análisis usando librerías profesionales (Pandas, NumPy, Scikit-learn), revisa el directorio **`Clase_02/Analisis_02/`**.

### Ejemplo Comparativo

**Calcular promedio de precios:**

```python
# Manual (Analisis_01/main.py)
suma = 0
for registro in datos:
    suma += float(registro['Price_USD'])
promedio = suma / len(datos)

# Profesional (Analisis_02/main.py)
promedio = df['Price_USD'].mean()
```

**Diferencias:**
- **Manual:** 4 líneas, ~0.5 segundos, educativo
- **Profesional:** 1 línea, ~0.01 segundos, eficiente

**Recomendación:** Aprende primero el enfoque manual (Analisis_01), luego pasa al profesional (Analisis_02).

---

## 🎓 Recomendaciones de Uso

### Para Estudiantes Principiantes
1. **Empieza aquí (Analisis_01)** para entender los fundamentos
2. Lee el código de `main.py` línea por línea
3. Ejecuta el script y observa los resultados
4. Intenta modificar las funciones para entender cómo funcionan
5. Luego pasa a **Analisis_02** para ver el enfoque profesional

### Para Estudiantes Intermedios
1. Compara `Analisis_01/main.py` con `Analisis_02/main.py`
2. Analiza las diferencias de rendimiento
3. Implementa tus propias funciones estadísticas
4. Experimenta con los ejercicios propuestos

### Para Instructores
1. Usa **Analisis_01** para enseñar fundamentos de programación
2. Muestra cómo funcionan los algoritmos internamente
3. Usa **Analisis_02** para mostrar mejores prácticas profesionales
4. Compara tiempos de ejecución en clase
5. Asigna ejercicios de ambos directorios

---

## 📖 Recursos Adicionales

### Documentación de Python
- [Módulo csv](https://docs.python.org/3/library/csv.html) - Lectura y escritura de archivos CSV
- [Módulo collections](https://docs.python.org/3/library/collections.html) - Counter, defaultdict
- [Módulo datetime](https://docs.python.org/3/library/datetime.html) - Manejo de fechas

### Conceptos Estadísticos
- **Media vs Mediana vs Moda** - Medidas de tendencia central
- **Desviación Estándar** - Medida de dispersión
- **Percentiles** - Valores que dividen la distribución
- **Correlación** - Relación entre variables
- **Outliers** - Valores atípicos (método IQR)
- **Análisis ABC** - Principio de Pareto (80/20)

### Próximos Pasos
1. **Completa este análisis manual** (Analisis_01)
2. **Revisa Analisis_02** para ver el enfoque profesional con librerías
3. **Explora Analisis_03** para análisis avanzado de calidad de agua

---

## 🤝 Contribuciones

Este material es parte de un curso de Computación. Si encuentras errores o tienes sugerencias, por favor repórtalos.

---

## 📝 Licencia

Material educativo de uso libre para fines académicos.

---

## ✨ Autor

David Marambio Salazar - 2025

---

## 🚀 ¡Comienza Ahora!

```bash
cd Clase_02/Analisis_01
python main.py
```

**¡Feliz aprendizaje! 📊🐍**

