# 📊 Análisis 02: Análisis Profesional con Librerías

## 📚 Descripción General

Este directorio contiene un sistema completo de análisis de datos implementado con **librerías profesionales de Python** (Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn).

**Propósito:** Mostrar cómo realizar análisis de datos de nivel profesional/industrial usando las herramientas estándar de la industria.

---

## 🎯 Objetivos de Aprendizaje

- ✅ Usar librerías estándar de la industria (Pandas, NumPy, Scikit-learn)
- ✅ Escribir código más eficiente y mantenible
- ✅ Aplicar técnicas de machine learning
- ✅ Crear visualizaciones profesionales
- ✅ Realizar análisis estadísticos avanzados
- ✅ Implementar pipelines de análisis de datos

---

## 📁 Estructura del Proyecto

```
Clase_02/Analisis_02/
│
├── 📂 sample/
│   └── BMW sales data (2010-2024) (1).csv  # Dataset de 50,000 registros
│
├── 🐍 main.py                              # ⭐ Sistema completo (2,100 líneas)
│
└── 📄 README.md                            # Este archivo
```

**Total:** 2,100 líneas de código | 50,000 registros | 11 columnas | 15 años de datos

## 📋 Características del Sistema

- ✅ **2,100 líneas** de código profesional
- ✅ **Librerías profesionales** (Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn)
- ✅ **Machine Learning** implementado (K-Means, PCA, Regresión Lineal)
- ✅ **Visualizaciones profesionales** (6 gráficos PNG generados)
- ✅ **Pruebas estadísticas** avanzadas (Shapiro-Wilk, Kruskal-Wallis, Chi-cuadrado)
- ✅ **Código optimizado** y eficiente (10-100x más rápido que implementación manual)

---

## 📚 Librerías Utilizadas

| Librería | Propósito | Funcionalidades Clave |
|----------|-----------|----------------------|
| **pandas** | Manipulación de datos | DataFrames, groupby, pivot tables, merge |
| **numpy** | Computación numérica | Arrays, operaciones vectorizadas, álgebra lineal |
| **scipy** | Estadística avanzada | Pruebas de hipótesis, distribuciones, tests estadísticos |
| **matplotlib** | Visualizaciones básicas | Gráficos de líneas, barras, scatter plots |
| **seaborn** | Gráficos estadísticos | Heatmaps, boxplots, distribuciones, KDE |
| **scikit-learn** | Machine Learning | Clustering, PCA, regresión, preprocesamiento |

---

## 🚀 Instalación y Ejecución

### Paso 1: Instalar Dependencias

#### Opción 1: Instalación Rápida
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

#### Opción 2: Versiones Específicas (Recomendado)
```bash
pip install numpy==1.26.4 pandas==2.2.3 scipy==1.15.2 matplotlib==3.9.2 seaborn==0.13.2 scikit-learn==1.5.1
```

#### Solución de Problemas con NumPy 2.x
Si encuentras errores de compatibilidad:
```bash
pip install numpy<2.0
pip install --upgrade pandas scipy matplotlib seaborn scikit-learn
```

### Paso 2: Ejecutar el Análisis

```bash
cd Clase_02/Analisis_02
python main.py
```

El script ejecutará automáticamente todos los niveles de análisis y generará visualizaciones.

---

## 📊 Niveles de Análisis Implementados

### 📊 Nivel 1: Estadística Básica (Pandas & NumPy)

**Funcionalidades:**
- **Carga eficiente** - `pd.read_csv()` con manejo automático de tipos
- **Estadísticas descriptivas** - `.describe()`, `.info()`, `.value_counts()`
- **Agrupaciones** - `.groupby()` con múltiples agregaciones
- **Operaciones vectorizadas** - Cálculos en columnas completas
- **Tablas cruzadas** - `pd.crosstab()` para análisis categórico
- **Matrices de correlación** - `.corr()` con método Pearson

**Ejemplo de código:**
```python
# Cargar datos
df = pd.read_csv('sample/BMW sales data (2010-2024) (1).csv')

# Estadísticas descriptivas
print(df.describe())

# Agrupación
precio_promedio = df.groupby('Model')['Price_USD'].mean()

# Correlación
correlacion = df[['Price_USD', 'Mileage_KM', 'Engine_Size_L']].corr()
```

### 📈 Nivel 2: Estadística Avanzada (SciPy)

**Funcionalidades:**
- **Test de normalidad** - Shapiro-Wilk para verificar distribución normal
- **Test de Kruskal-Wallis** - Comparación de múltiples grupos
- **Intervalos de confianza** - Estimación de parámetros poblacionales
- **Asimetría y curtosis** - Forma de la distribución
- **Test Chi-cuadrado** - Independencia entre variables categóricas
- **Correlación de Spearman** - Correlación no paramétrica
- **Análisis de percentiles** - Cuartiles y percentiles personalizados

**Ejemplo de código:**
```python
from scipy import stats

# Test de normalidad
statistic, p_value = stats.shapiro(df['Price_USD'])

# Test de Kruskal-Wallis
h_stat, p_val = stats.kruskal(*[group['Price_USD'] for name, group in df.groupby('Fuel_Type')])

# Correlación de Spearman
corr, p_value = stats.spearmanr(df['Price_USD'], df['Mileage_KM'])
```

### 🔬 Nivel 3: Ciencia de Datos (Scikit-Learn)

**Funcionalidades:**
- **K-Means Clustering** - Segmentación automática de clientes
- **PCA** - Reducción de dimensionalidad y visualización
- **Regresión Lineal** - Predicción de precios y volumen de ventas
- **Detección de outliers** - Método IQR robusto
- **Análisis temporal** - Tendencias y evolución
- **Preprocesamiento** - StandardScaler, LabelEncoder

**Ejemplo de código:**
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Regresión Lineal
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 🎨 Visualizaciones Profesionales

El script genera automáticamente **6 gráficos en formato PNG**:

1. **`distribucion_precios.png`** - Histograma con KDE (Kernel Density Estimation)
2. **`precios_por_combustible.png`** - Box plots comparativos por tipo de combustible
3. **`matriz_correlacion.png`** - Heatmap de correlaciones entre variables numéricas
4. **`precio_vs_kilometraje.png`** - Scatter plot con línea de regresión
5. **`ventas_por_region.png`** - Gráfico de barras por región
6. **`tendencia_temporal.png`** - Serie temporal de ventas (2010-2024)

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

### 🟢 Nivel Básico - Pandas & NumPy
1. Usa `df.groupby()` para calcular precio promedio por modelo
2. Filtra vehículos recientes con `df[df['Year'] > 2020]`
3. Crea columna `Price_per_KM` usando operaciones vectorizadas
4. Usa `pd.crosstab()` para analizar Region × Fuel_Type
5. Calcula percentiles con `np.percentile()` o `df.quantile()`
6. Encuentra el modelo más vendido por región

### 🟡 Nivel Intermedio - SciPy
1. Realiza test t de Student entre precios de dos regiones
2. Calcula intervalos de confianza al 95% para precios
3. Usa `stats.pearsonr()` para correlación con p-valores
4. Aplica test de normalidad (Shapiro-Wilk) a precios
5. Realiza ANOVA entre grupos de combustible
6. Calcula correlación de Spearman entre variables ordinales

### 🔴 Nivel Avanzado - Machine Learning
1. Entrena regresión lineal para predecir `Sales_Volume`
2. Aplica K-Means con elbow method para encontrar k óptimo
3. Usa PCA para reducir dimensionalidad y visualiza en 2D
4. Implementa validación cruzada para evaluar modelos
5. Crea pipeline de preprocesamiento con StandardScaler
6. Compara múltiples modelos de regresión (Linear, Ridge, Lasso)

---

## 💡 Ventajas del Enfoque Profesional

### ✅ Eficiencia y Rendimiento
- **10-100x más rápido** - Operaciones vectorizadas optimizadas en C
- **Manejo de grandes datasets** - Procesamiento eficiente de millones de registros
- **Uso óptimo de memoria** - Gestión automática de recursos

### ✅ Productividad
- **Código más corto** - 1 línea vs 10+ líneas de código manual
- **Más legible** - Sintaxis clara y expresiva
- **Menos errores** - Funciones probadas y validadas

### ✅ Capacidades Avanzadas
- **Machine Learning** - Algoritmos listos para usar
- **Visualizaciones profesionales** - Gráficos de calidad publicable
- **Pruebas estadísticas** - Tests avanzados implementados
- **Estándar de la industria** - Usado en empresas y academia

### ✅ Ecosistema Rico
- **Documentación extensa** - Tutoriales, ejemplos, comunidad
- **Integración** - Compatible con otras herramientas (Jupyter, SQL, etc.)
- **Actualizaciones constantes** - Mejoras y nuevas funcionalidades

---

## 🔄 Comparación con Análisis Manual

Si quieres entender cómo funcionan los algoritmos internamente antes de usar librerías, revisa el directorio **`Clase_02/Analisis_01/`**.

### Ejemplo Comparativo

**Calcular promedio de precios:**

```python
# Manual (Analisis_01/main.py)
suma = 0
for registro in datos:
    suma += float(registro['Price_USD'])
promedio = suma / len(datos)
# Tiempo: ~0.5 segundos para 50,000 registros

# Profesional (Analisis_02/main.py)
promedio = df['Price_USD'].mean()
# Tiempo: ~0.01 segundos para 50,000 registros
```

**Diferencias:**
- **Manual:** 4 líneas, ~0.5s, educativo, entiendes el algoritmo
- **Profesional:** 1 línea, ~0.01s, eficiente, estándar de la industria

**Recomendación:** Aprende primero el enfoque manual (Analisis_01) para entender los fundamentos, luego usa el profesional (Analisis_02) para proyectos reales.

---

## 🎓 Recomendaciones de Uso

### Para Estudiantes Principiantes
1. **Primero completa Analisis_01** para entender los fundamentos
2. Lee el código de `main.py` para ver cómo se usan las librerías
3. Ejecuta el script y observa las visualizaciones generadas
4. Compara con tu implementación manual de Analisis_01
5. Experimenta modificando parámetros y funciones

### Para Estudiantes Intermedios
1. Compara `Analisis_01/main.py` con `Analisis_02/main.py`
2. Analiza las diferencias de rendimiento (usa `time.time()`)
3. Implementa los ejercicios propuestos usando Pandas/NumPy
4. Crea tus propias visualizaciones personalizadas
5. Explora la documentación de las librerías

### Para Estudiantes Avanzados
1. Implementa pipelines de machine learning completos
2. Optimiza el código para datasets más grandes
3. Crea dashboards interactivos con Plotly o Streamlit
4. Integra con bases de datos (SQL)
5. Despliega modelos en producción

### Para Instructores
1. Usa **Analisis_01** para enseñar fundamentos de programación
2. Usa **Analisis_02** para mostrar mejores prácticas profesionales
3. Compara tiempos de ejecución en clase (manual vs librerías)
4. Muestra las visualizaciones generadas
5. Asigna proyectos que combinen ambos enfoques

---

## 📖 Recursos Adicionales

### Documentación Oficial
- **[Pandas](https://pandas.pydata.org/docs/)** - Manipulación y análisis de datos
- **[NumPy](https://numpy.org/doc/)** - Computación numérica
- **[SciPy](https://docs.scipy.org/doc/scipy/)** - Algoritmos científicos
- **[Matplotlib](https://matplotlib.org/stable/contents.html)** - Visualizaciones
- **[Seaborn](https://seaborn.pydata.org/)** - Gráficos estadísticos
- **[Scikit-Learn](https://scikit-learn.org/stable/)** - Machine Learning

### Tutoriales Recomendados
- **[Pandas Tutorial](https://pandas.pydata.org/pandas-docs/stable/getting_started/intro_tutorials/)** - Introducción a Pandas
- **[NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)** - Guía rápida de NumPy
- **[Scikit-Learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)** - Tutoriales de ML
- **[Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)** - Galería de visualizaciones

### Libros Recomendados
- **"Python for Data Analysis"** - Wes McKinney (creador de Pandas)
- **"Hands-On Machine Learning"** - Aurélien Géron
- **"Python Data Science Handbook"** - Jake VanderPlas

### Próximos Pasos
1. **Completa este análisis profesional** (Analisis_02)
2. **Compara con Analisis_01** para ver las diferencias
3. **Explora Analisis_03** para análisis avanzado de calidad de agua
4. **Crea tu propio proyecto** aplicando lo aprendido

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
# Instalar dependencias
pip install pandas numpy scipy matplotlib seaborn scikit-learn

# Ejecutar análisis
cd Clase_02/Analisis_02
python main.py
```

**¡Feliz aprendizaje con librerías profesionales! 📊🐍🚀**

