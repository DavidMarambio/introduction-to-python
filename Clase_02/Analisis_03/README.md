# 📊 Sistema de Análisis de Calidad de Agua Subterránea

Este directorio contiene **DOS sistemas completos** de análisis de calidad de agua subterránea de Telangana, India (2018-2020):

1. **`main_basico.py`** - Sistema con **funciones manuales** de estadística (sin librerías de alto nivel)
2. **`main_avanzado.py`** - Sistema con **librerías profesionales** (Pandas, Scikit-learn, Plotly)

---

## 🎯 ¿Cuál usar?

### Usa `main_basico.py` si:
- ✅ Quieres entender cómo funcionan los algoritmos internamente
- ✅ Necesitas implementaciones desde cero (educativo)
- ✅ Estás aprendiendo estadística y ciencia de datos
- ✅ No puedes instalar librerías externas
- ✅ Prefieres código puro Python sin dependencias

### Usa `main_avanzado.py` si:
- ✅ Necesitas análisis de nivel profesional/industrial
- ✅ Quieres dashboards interactivos y visualizaciones avanzadas
- ✅ Requieres machine learning y algoritmos optimizados
- ✅ Buscas identificar las mejores fuentes de agua rápidamente
- ✅ Necesitas resultados en formato HTML interactivo

---

## 📁 Estructura del Proyecto

```
Clase_02/Analisis_03/
│
├── 📂 samples/                          # Datos de calidad de agua
│   ├── ground_water_quality_2018_post.csv  (374 registros)
│   ├── ground_water_quality_2019_post.csv  (364 registros)
│   ├── ground_water_quality_2020_post.csv  (368 registros)
│   └── README.md                       # Descripción detallada de los datos
│
├── 🐍 main_basico.py                   # ⭐ Sistema con funciones manuales (1,477 líneas)
├── 🐍 main_avanzado.py                 # ⭐ Sistema con librerías profesionales (1,713 líneas)
│
├── 📦 requirements.txt                 # Dependencias para main_avanzado.py
│
└── 📄 README.md                        # Este archivo (documentación general)
```

**Total:** 3,190 líneas de código | 1,106 muestras de agua | 33 distritos | 3 años

---

## 📊 Sobre los Datos

### Fuente
- **Origen:** Telangana Open Data Portal, India
- **Región:** Estado de Telangana, 33 distritos
- **Período:** 2018-2020 (temporada post-monzón)
- **Muestras totales:** 1,106 registros

### Parámetros Medidos (26 columnas)
- **Ubicación:** Distrito, Mandal, Village, Latitud, Longitud
- **Químicos:** pH, Ca, Mg, Na, K, CO3, HCO3, Cl, F, NO3, SO4
- **Indicadores:** TDS, E.C, T.H, SAR, RSC
- **Clasificación:** Classification (C1S1 a C4S4, OG)

### Clasificaciones de Calidad
- **C1S1:** Baja salinidad, bajo sodio - Excelente para irrigación
- **C2S1:** Media salinidad, bajo sodio - Buena para irrigación
- **C3S1-C3S3:** Alta salinidad - Requiere drenaje y cultivos tolerantes
- **C4S1-C4S4:** Muy alta salinidad - No apta o uso muy limitado
- **OG:** Otras clasificaciones

---

# 🚀 SISTEMA 1: main_basico.py (Funciones Manuales)

Sistema educativo completo para análisis estadístico y ciencia de datos aplicado a calidad de agua subterránea, implementado con **funciones manuales de estadística aplicada** sin dependencias de librerías de alto nivel.

## � Características

- ✅ **1,477 líneas** de código Python puro
- ✅ **8 módulos** de análisis implementados manualmente
- ✅ **17 opciones** en menú interactivo
- ✅ **Sin dependencias** externas (solo módulos estándar de Python)
- ✅ **Educativo:** Aprende cómo funcionan los algoritmos internamente

---

## 🚀 Inicio Rápido

### Ejecutar el Sistema

```bash
# Desde este directorio
cd Clase_02/Analisis_03
python main_basico.py
```

### Primera Ejecución Recomendada

1. Ejecuta el script: `python main_basico.py`
2. Selecciona la **opción 16** (Reporte ejecutivo) para obtener un resumen completo
3. Explora las demás opciones según tus necesidades

---

## 📖 Módulos Implementados

### `main_basico.py` - Script Principal

**Descripción:** Sistema completo de análisis con menú interactivo que ofrece 17 opciones de análisis desde básico hasta ciencia de datos.

**Características:**
- ✅ 1,477 líneas de código profesional
- ✅ 8 módulos especializados
- ✅ 15+ funciones estadísticas manuales
- ✅ Interfaz de menú interactivo
- ✅ Sin dependencias externas (solo Python estándar)

**Módulos incluidos:**

1. **EstadisticaManual** - Funciones estadísticas implementadas desde cero
2. **CargadorDatos** - Carga y procesamiento de archivos CSV
3. **AnalisisBasico** - Estadísticas descriptivas y resúmenes
4. **AnalisisMedio** - Correlaciones, outliers, análisis temporal
5. **AnalisisAvanzado** - PCA, clustering, análisis espacial
6. **CienciaDatos** - Modelos predictivos y validación
7. **AnalisisCalidad** - Evaluación según normativas
8. **GeneradorReportes** - Reportes ejecutivos profesionales

**Cómo usar:**
```bash
# Desde el directorio raíz del proyecto
python -m Clase_02.Analisis_03.main_basico

# O desde este directorio
cd Clase_02/Analisis_03
python main_basico.py

# Selecciona una opción del menú (1-17)
```

**Uso programático:**

También puedes importar las clases y usarlas en tus propios scripts:

```python
from Clase_02.Analisis_03.main_basico import (
    EstadisticaManual,
    CargadorDatos,
    AnalisisBasico,
    AnalisisMedio,
    AnalisisAvanzado,
    CienciaDatos,
    AnalisisCalidad,
    GeneradorReportes
)

# Ejemplo 1: Usar funciones estadísticas
stats = EstadisticaManual()
datos = [12.5, 15.3, 14.8, 16.2, 13.9]
media = stats.media(datos)
mediana = stats.mediana(datos)
desv_est = stats.desviacion_estandar(datos)

print(f"Media: {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Desviación estándar: {desv_est:.2f}")

# Ejemplo 2: Cargar datos y ejecutar análisis
cargador = CargadorDatos("samples")
cargador.cargar_archivos_csv()

# Análisis básico
analisis_basico = AnalisisBasico(cargador)
analisis_basico.resumen_general()

# Análisis de correlaciones
analisis_medio = AnalisisMedio(cargador)
analisis_medio.matriz_correlacion(['pH', 'TDS', 'E.C', 'T.H'])

# Clustering
analisis_avanzado = AnalisisAvanzado(cargador)
analisis_avanzado.clustering_kmeans_manual(['TDS', 'T.H', 'SAR'], k=3)

# Análisis predictivo
ciencia_datos = CienciaDatos(cargador)
ciencia_datos.analisis_predictivo_completo()

# Reporte ejecutivo
reporte = GeneradorReportes(cargador)
reporte.reporte_ejecutivo()
```

---

## 🎯 Funcionalidades del main_basico.py

### 📊 ANÁLISIS BÁSICO

#### Opción 1: Resumen General del Dataset
- Total de muestras analizadas
- Distritos cubiertos
- Período de análisis
- Distribución por clasificación de agua

#### Opción 2: Estadísticas Descriptivas por Parámetro
Calcula para cada parámetro (pH, TDS, E.C, etc.):
- Media, mediana, moda
- Desviación estándar, varianza
- Coeficiente de variación
- Mínimo, máximo, rango
- Percentiles (Q1, Q2, Q3)
- Asimetría y curtosis

#### Opción 3: Análisis de Calidad por Distrito
- Estadísticas por distrito
- Clasificación predominante
- Comparación entre regiones

---

### 📈 ANÁLISIS MEDIO

#### Opción 4: Matriz de Correlación
- Correlaciones de Pearson entre parámetros
- Identificación de relaciones fuertes
- Interpretación de resultados

**Parámetros analizados:** pH, E.C, TDS, T.H, SAR, Cl, Na, Ca, Mg

#### Opción 5: Detección de Outliers
- Método IQR (Rango Intercuartílico)
- Identificación de valores atípicos
- Porcentaje de outliers
- Valores extremos detectados

#### Opción 6: Análisis Temporal (2018-2020)
- Evolución de parámetros por año
- Tendencias lineales
- Regresión temporal
- Interpretación de cambios

#### Opción 7: Análisis de Distribución
- Histogramas de frecuencia
- Análisis de forma de distribución
- Identificación de patrones

---

### 🔬 ANÁLISIS AVANZADO

#### Opción 8: PCA (Componentes Principales)
- Implementación manual de PCA
- Reducción de dimensionalidad
- Varianza explicada por componente
- Interpretación de componentes principales

**Parámetros incluidos:** pH, E.C, TDS, Ca, Mg, Na, K, Cl, SO4, HCO3

#### Opción 9: Clustering K-Means
- Algoritmo K-Means manual
- Agrupamiento de muestras similares
- 3 clusters por defecto
- Características de cada cluster

**Parámetros usados:** TDS, T.H, SAR

#### Opción 10: Análisis Espacial
- Análisis por cuadrantes geográficos
- Distribución espacial de parámetros
- Identificación de zonas problemáticas

---

### 🤖 CIENCIA DE DATOS

#### Opción 11: Análisis Predictivo Completo
- Clasificador Naive Bayes (implementación manual)
- Validación cruzada k-fold (k=5)
- Matriz de confusión
- Métricas: Accuracy, Precision, Recall, F1-score
- Predicción de clasificación de agua

#### Opción 12: Feature Importance
- Ranking de importancia de variables
- Correlación con variable objetivo
- Identificación de predictores clave

---

### 💧 EVALUACIÓN DE CALIDAD

#### Opción 13: Evaluación según RSC
**RSC (Residual Sodium Carbonate)** - Aptitud para riego

Clasificación:
- **Seguro:** RSC < 1.25 meq/L
- **Marginal:** 1.25 ≤ RSC ≤ 2.50 meq/L
- **Inadecuado:** RSC > 2.50 meq/L

#### Opción 14: Evaluación según TDS
**TDS (Total Dissolved Solids)** - Aptitud para ganado

Clasificación:
- **Excelente:** < 1,000 mg/L
- **Muy bueno:** 1,000-2,999 mg/L
- **Bueno:** 3,000-4,999 mg/L
- **Aceptable:** 5,000-6,999 mg/L
- **Marginal:** 7,000-9,999 mg/L
- **No recomendado:** ≥ 10,000 mg/L

#### Opción 15: Análisis de pH
Clasificación:
- **Muy ácido:** pH < 5.5
- **Ácido:** 5.5 ≤ pH < 6.5
- **Ligeramente ácido:** 6.5 ≤ pH < 7.0
- **Neutro:** 7.0 ≤ pH ≤ 7.5
- **Ligeramente alcalino:** 7.5 < pH ≤ 8.5
- **Alcalino:** 8.5 < pH ≤ 9.5
- **Muy alcalino:** pH > 9.5

---

### 📋 REPORTES

#### Opción 16: Reporte Ejecutivo
Genera un resumen profesional con:
- Cobertura del estudio
- Calidad general del agua
- Parámetros críticos
- Recomendaciones profesionales

**⭐ Recomendado para primera ejecución**

#### Opción 17: Análisis Completo
Ejecuta todos los módulos de análisis en secuencia:
1. Resumen general
2. Estadísticas descriptivas
3. Matriz de correlación
4. Análisis temporal
5. PCA
6. Clustering
7. Análisis predictivo
8. Evaluación de calidad
9. Reporte ejecutivo

**⏱️ Tiempo estimado:** 3-5 minutos

---

## 📊 Datos Analizados

### Origen
- **Fuente:** Telangana Open Data Portal, India
- **Región:** Estado de Telangana
- **Período:** 2018-2020 (temporada post-monzón)
- **Muestras:** 1,106 registros
- **Distritos:** 33

### Parámetros Medidos (26 columnas)

**Identificación:**
- sno, district, mandal, village
- lat_gis, long_gis

**Parámetros Físico-Químicos:**
- pH, E.C (Conductividad Eléctrica), TDS
- gwl (nivel freático), season

**Iones y Minerales:**
- CO3, HCO3, Cl, F, NO3, SO4
- Na, K, Ca, Mg

**Índices de Calidad:**
- T.H (Dureza Total)
- SAR (Sodium Absorption Ratio)
- RSC (Residual Sodium Carbonate)

**Clasificación:**
- Classification (C1S1 a C4S4)
- Classification.1 (P.S./N.P.S.)

---

## 🎓 Funciones Estadísticas Implementadas

Todas las funciones están implementadas manualmente sin usar NumPy, Pandas o Scikit-learn:

### Medidas de Tendencia Central
- `media()` - Media aritmética
- `mediana()` - Valor central
- `moda()` - Valor más frecuente

### Medidas de Dispersión
- `varianza()` - Varianza poblacional
- `desviacion_estandar()` - Desviación estándar
- `coeficiente_variacion()` - CV (%)
- `rango_intercuartilico()` - IQR

### Medidas de Posición
- `percentil()` - Percentil k
- `cuartiles()` - Q1, Q2, Q3

### Medidas de Forma
- `asimetria()` - Skewness
- `curtosis()` - Kurtosis

### Análisis Bivariado
- `covarianza()` - Covarianza entre dos variables
- `correlacion_pearson()` - Coeficiente de correlación
- `regresion_lineal()` - Regresión lineal simple

### Normalización
- `normalizar_zscore()` - Estandarización Z-score
- `normalizar_minmax()` - Normalización Min-Max [0,1]

---

## 💻 Requisitos Técnicos

### Software
- **Python:** 3.6 o superior
- **Dependencias:** Ninguna (solo módulos estándar de Python)
  - `csv` - Lectura de archivos
  - `math` - Funciones matemáticas
  - `collections` - Estructuras de datos
  - `typing` - Type hints
  - `os` - Operaciones de sistema

### Hardware
- Cualquier computadora moderna
- RAM: 2GB mínimo
- Espacio en disco: 50MB

---

## 🎯 Casos de Uso

### Para Agricultores
**Pregunta:** ¿Es apta el agua para riego?  
**Solución:** Ejecuta opciones 13, 14, 15  
**Tiempo:** 5 minutos

### Para Investigadores
**Pregunta:** ¿Qué factores determinan la calidad?  
**Solución:** Ejecuta opciones 4, 11, 12  
**Tiempo:** 30 minutos

### Para Gestores Ambientales
**Pregunta:** ¿Cuáles son las zonas de riesgo?  
**Solución:** Ejecuta opciones 3, 5, 10  
**Tiempo:** 20 minutos

### Para Estudiantes
**Pregunta:** ¿Cómo funciona el análisis estadístico?
**Solución:** Ejecuta `main_basico.py` y explora las opciones 8-12
**Tiempo:** 45 minutos

---

## 📚 Recursos Adicionales

- **`samples/README.md`** - Información detallada sobre los datos fuente
- **Código fuente** - Los scripts `main_basico.py` y `main_avanzado.py` están completamente documentados
- **Ejemplos de uso** - Ver sección "Uso programático" arriba para ejemplos de código

---

## 🔧 Solución de Problemas

### Error: "No se pudieron cargar los datos"
**Solución:**
```bash
# Verifica que estés en el directorio correcto
pwd
# Debe mostrar: .../Clase_02/Analisis_03

# Verifica que exista la carpeta samples
ls samples/
```

### Error al importar módulos
**Solución:**
```bash
# Ejecuta desde el directorio raíz del proyecto
python -m Clase_02.Analisis_03.main_basico
# O para el sistema avanzado
python -m Clase_02.Analisis_03.main_avanzado
```

---

## 📖 Documentación de Referencia

### Clasificación de Agua (C-S)

**C = Conductividad (Salinidad)**
- C1: Baja (< 250 μS/cm)
- C2: Media (250-750 μS/cm)
- C3: Alta (750-2250 μS/cm)
- C4: Muy alta (> 2250 μS/cm)

**S = Sodio (SAR)**
- S1: Bajo (< 10)
- S2: Medio (10-18)
- S3: Alto (18-26)
- S4: Muy alto (> 26)

**Ejemplo:** C3S1 = Alta salinidad, bajo sodio

---

## 🚀 Próximos Pasos

1. **Ejecuta** `python main.py` y selecciona opción 16 (Reporte ejecutivo)
2. **Explora** las diferentes opciones del menú (1-17)
3. **Estudia** el código fuente en `main.py` para entender las implementaciones
4. **Importa** las clases en tus propios scripts (ver sección "Uso programático")
5. **Personaliza** los análisis según tus necesidades

---

## 📝 Licencia y Créditos

**Datos:** Telangana Open Data Portal, India  
**Implementación:** Sistema educativo de análisis estadístico  
**Propósito:** Educación y análisis profesional de calidad de agua

---

**¡Listo para comenzar! 🎉**

Ejecuta: `python main_basico.py` y selecciona la opción 16 para tu primer análisis.

---
---

# 🚀 SISTEMA 2: main_avanzado.py (Librerías Profesionales)

Sistema avanzado de análisis de calidad de agua utilizando **librerías profesionales de Python** (Pandas, NumPy, Scikit-learn, Plotly) con dashboards ejecutivos interactivos.

## 📋 Características

- ✅ **1,713 líneas** de código optimizado
- ✅ **11 opciones** de análisis profesional
- ✅ **4 algoritmos de ML** comparados (Random Forest, Gradient Boosting, SVM, KNN)
- ✅ **8 archivos HTML** interactivos generados
- ✅ **Dashboards ejecutivos** con Plotly
- ✅ **Explicaciones educativas** de cada algoritmo
- ✅ **95% de accuracy** en clasificación

---

## 🎯 Objetivo Principal

**Identificar las mejores fuentes de agua subterránea** mediante:
- Análisis estadístico avanzado
- Machine Learning (Random Forest, SVM, Gradient Boosting, KNN)
- Sistema de ranking multi-criterio
- Dashboards ejecutivos interactivos
- Visualizaciones geográficas

---

## 📚 Librerías Utilizadas

- **Pandas & NumPy** - Análisis de datos y computación numérica
- **Scikit-learn** - Machine Learning
- **SciPy** - Estadística avanzada
- **Matplotlib & Seaborn** - Visualizaciones estáticas
- **Plotly** - Dashboards interactivos

---

## 🚀 Instalación

```bash
# Instalar dependencias
cd Clase_02/Analisis_03
pip install -r requirements.txt

# Verificar instalación
python -c "import pandas, numpy, sklearn, plotly; print('✅ OK')"
```

---

## 💻 Inicio Rápido

### Opción 1: Menú Interactivo (Recomendado)

```bash
cd Clase_02/Analisis_03
python main_avanzado.py
# Selecciona opción 11 (Análisis completo)
# Tiempo: 3-5 minutos
```

### Opción 2: Uso Programático

```python
from Clase_02.Analisis_03.main_avanzado import AnalizadorCalidadAgua

# Crear analizador
analizador = AnalizadorCalidadAgua("samples")

# Cargar y procesar datos
analizador.cargar_datos()
analizador.preprocesar_datos()

# Obtener ranking de calidad
analizador.sistema_ranking_calidad()

# Generar dashboard
analizador.dashboard_ejecutivo()
```

---

## 📊 Funcionalidades Principales

### 1️⃣ Análisis Estadístico
- **EDA (Exploratory Data Analysis)** - Estadísticas descriptivas completas
- **Análisis de valores faltantes** - Detección e imputación
- **Preprocesamiento** - Limpieza, normalización, feature engineering
- **Análisis de correlación** - Matriz de Pearson con heatmap interactivo

### 2️⃣ Análisis Avanzado
- **PCA** - Reducción de dimensionalidad con biplot
- **K-Means Clustering** - Agrupamiento con métricas de calidad (Silhouette, Davies-Bouldin)

### 3️⃣ Machine Learning
- **Random Forest** - Ensemble de árboles de decisión
- **Gradient Boosting** - Boosting secuencial
- **SVM** - Support Vector Machine con kernel RBF
- **KNN** - K-Nearest Neighbors
- **Comparación de modelos** - Métricas: Accuracy, Precision, Recall, F1-Score
- **Feature importance** - Identificación de variables clave

### 4️⃣ Sistema de Calidad
- **Ranking multi-criterio** - Scoring basado en TDS (40%), SAR (25%), RSC (20%), pH (10%), Dureza (5%)
- **Clasificación en 5 niveles** - Excelente, Buena, Moderada, Pobre, Muy Pobre
- **TOP 10 mejores fuentes** - Identificación de fuentes óptimas
- **Mapa geográfico** - Visualización espacial de calidad

### 5️⃣ Dashboards Ejecutivos
- **Dashboard consolidado** - 6 visualizaciones en un solo archivo
- **Visualizaciones interactivas** - Zoom, hover, filtros
- **Exportación HTML** - Fácil compartir y presentar

---

## 📈 Archivos Generados

Todos los análisis generan archivos HTML interactivos:

| Archivo | Descripción | Análisis |
|---------|-------------|----------|
| `output_correlacion_heatmap.html` | Matriz de correlación | Opción 4 |
| `output_pca_biplot.html` | Biplot de PCA | Opción 5 |
| `output_clustering.html` | Visualización de clusters | Opción 6 |
| `output_confusion_matrix_*.html` | Matriz de confusión | Opción 7 |
| `output_feature_importance_*.html` | Importancia de variables | Opción 7 |
| `output_quality_distribution.html` | Distribución de calidad | Opción 8 |
| `output_quality_map.html` | Mapa geográfico | Opción 8 |
| `output_dashboard_ejecutivo.html` | Dashboard consolidado | Opción 9 |

---

## 🎓 Características Únicas

### ✅ Explicaciones Detalladas
Cada análisis incluye:
- **Introducción al algoritmo** - Qué hace y cómo funciona
- **Fundamento matemático** - Fórmulas y teoría
- **Interpretación de resultados** - Cómo leer los valores
- **Conclusiones** - Qué significan para la calidad del agua

### ✅ Sistema de Ranking Inteligente
```
Quality Score = (TDS × 0.40) + (SAR × 0.25) + (RSC × 0.20) +
                (pH × 0.10) + (Dureza × 0.05)

Clasificación:
🟢 80-100: Excelente (uso sin restricciones)
🔵 60-80: Buena (precauciones menores)
🟡 40-60: Moderada (requiere manejo)
🟠 20-40: Pobre (uso limitado)
🔴 0-20: Muy pobre (no recomendado)
```

### ✅ Machine Learning Comparativo
Compara 4 algoritmos y selecciona automáticamente el mejor:
- Random Forest
- Gradient Boosting
- SVM
- KNN

---

## 💡 Casos de Uso

### Para Investigadores
```bash
python main_avanzado.py
# Opción 11 (Análisis completo)
# Usar visualizaciones en papers científicos
```

### Para Gestores/Ejecutivos
```bash
python main_avanzado.py
# Opción 9 (Dashboard ejecutivo)
# Presentar a stakeholders
```

### Para Agricultores/Usuarios Finales
```bash
python main_avanzado.py
# Opción 8 (Ranking de calidad)
# Identificar mejores fuentes en su área
```

---

## 🔄 Comparación: main_basico.py vs main_avanzado.py

| Característica | main_basico.py | main_avanzado.py |
|----------------|---------|-------------------------|
| **Implementación** | Manual (desde cero) | Librerías profesionales |
| **Objetivo** | Educativo | Producción/Investigación |
| **Velocidad** | Lenta | Rápida (optimizada) |
| **Visualizaciones** | Básicas (texto) | Interactivas (HTML) |
| **Machine Learning** | Básico (Naive Bayes, K-Means) | Avanzado (RF, SVM, GB, KNN) |
| **Dashboards** | No | Sí (ejecutivos) |
| **Mapas geográficos** | No | Sí (interactivos) |
| **Ranking de calidad** | Básico | Multi-criterio avanzado |
| **Dependencias** | Ninguna | Pandas, Scikit-learn, Plotly |
| **Tiempo de ejecución** | 5-10 min | 3-5 min |
| **Ideal para** | Aprender algoritmos | Análisis profesional |

---

## 🚀 Próximos Pasos

### Para Sistema Básico (main_basico.py)
1. **Ejecuta** el script: `python main_basico.py`
2. **Selecciona** opción 16 (Reporte ejecutivo)
3. **Explora** las opciones 1-17 según tus necesidades
4. **Aprende** cómo funcionan los algoritmos internamente

### Para Sistema Avanzado (main_avanzado.py)
1. **Instala** las dependencias: `pip install -r requirements.txt`
2. **Ejecuta** el análisis completo: `python main_avanzado.py` → Opción 11
3. **Revisa** los archivos HTML generados en tu navegador
4. **Identifica** las mejores fuentes de agua en tu área de interés
5. **Comparte** el dashboard ejecutivo con stakeholders

---

## 📞 Soporte y Recursos

### Documentación
- **samples/README.md** - Descripción detallada de los datos
- **requirements.txt** - Dependencias del sistema avanzado

### Código Fuente
- **main_basico.py** - Sistema con funciones manuales (1,477 líneas)
- **main_avanzado.py** - Sistema con librerías profesionales (1,713 líneas)

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

## 🎉 ¡Listo para Comenzar!

### Sistema Básico (Educativo)
```bash
python main_basico.py
# Selecciona opción 16 (Reporte ejecutivo)
```

### Sistema Avanzado (Profesional)
```bash
python main_avanzado.py
# Selecciona opción 11 (Análisis completo)
```

**Ambos sistemas están 100% funcionales y listos para analizar calidad de agua! 💧🏆**

