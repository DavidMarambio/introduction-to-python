# 📊 Clase 02: Análisis de Datos con Python

## 📚 Descripción General

Esta clase está dedicada al **análisis de datos** en Python, cubriendo desde implementaciones manuales hasta el uso de librerías profesionales y machine learning.

Contiene **3 proyectos completos** que te enseñarán a analizar datos de diferentes maneras, desde los fundamentos hasta técnicas avanzadas.

---

## 📂 Estructura del Directorio

```
Clase_02/
│
├── 📁 Analisis_01/                     # Análisis Manual (sin librerías)
│   ├── main.py                         (2,157 líneas)
│   ├── sample/
│   │   └── BMW sales data (2010-2024) (1).csv
│   └── README.md
│
├── 📁 Analisis_02/                     # Análisis Profesional (con librerías)
│   ├── main.py                         (2,100 líneas)
│   ├── sample/
│   │   └── BMW sales data (2010-2024) (1).csv
│   └── README.md
│
├── 📁 Analisis_03/                     # Análisis de Calidad de Agua
│   ├── main_basico.py                  (1,476 líneas)
│   ├── main_avanzado.py                (1,712 líneas)
│   ├── samples/
│   │   └── [1,106 muestras de agua]
│   └── README.md
│
└── 📄 README.md                        # Este archivo
```

**Total:** 7,445 líneas de código | 3 proyectos | 2 datasets | 51,106 registros

---

## 🎯 Objetivos de Aprendizaje

Al completar esta clase, serás capaz de:

- ✅ Implementar algoritmos estadísticos desde cero
- ✅ Usar librerías profesionales (Pandas, NumPy, SciPy, Scikit-learn)
- ✅ Realizar análisis exploratorio de datos (EDA)
- ✅ Crear visualizaciones profesionales
- ✅ Aplicar técnicas de machine learning
- ✅ Comparar enfoques manuales vs profesionales
- ✅ Desarrollar proyectos completos de análisis de datos

---

## 📊 Proyectos Incluidos

### 📁 Analisis_01: Análisis Manual de Datos CSV

**🎯 Propósito:** Entender cómo funcionan los algoritmos estadísticos implementándolos desde cero.

#### Características
- 🐍 **2,157 líneas** de código Python puro
- ✅ **Sin dependencias** externas (solo módulos estándar: csv, os, datetime, collections)
- 📊 **25+ funciones** estadísticas implementadas manualmente
- 📈 **3 niveles** de análisis progresivo
- 🚗 **50,000 registros** de ventas BMW (2010-2024)

#### Niveles de Análisis
1. **📊 Nivel 1: Estadística Básica**
   - Lectura de CSV, conteo de registros
   - Promedios, sumas, mínimos, máximos
   - Frecuencias y distribuciones
   - Valores únicos

2. **📈 Nivel 2: Estadística Avanzada**
   - Mediana, moda, desviación estándar
   - Varianza, percentiles
   - Coeficiente de variación
   - Correlaciones numéricas

3. **🔬 Nivel 3: Ciencia de Datos**
   - Análisis temporal
   - Detección de outliers (método IQR)
   - Análisis ABC/Pareto (80/20)
   - Segmentación de datos
   - Tasas de crecimiento

#### Ejecución
```bash
cd Analisis_01
python main.py
```

#### Ideal Para
- 🎓 Estudiantes que quieren entender los fundamentos
- 💡 Aprender cómo funcionan los algoritmos internamente
- 🔧 Desarrollar habilidades de programación
- 📚 Preparación para usar librerías profesionales

---

### 📁 Analisis_02: Análisis Profesional con Librerías

**🎯 Propósito:** Aprender a usar las herramientas estándar de la industria para análisis de datos.

#### Características
- 🐍 **2,100 líneas** de código profesional
- 📚 **6 librerías** profesionales utilizadas
- 🤖 **Machine Learning** implementado
- 📊 **6 visualizaciones** PNG generadas automáticamente
- ⚡ **10-100x más rápido** que implementación manual
- 🚗 **50,000 registros** de ventas BMW (2010-2024)

#### Librerías Utilizadas
| Librería | Propósito |
|----------|-----------|
| **Pandas** | Manipulación de datos con DataFrames |
| **NumPy** | Computación numérica y arrays |
| **SciPy** | Estadística avanzada y tests |
| **Matplotlib** | Visualizaciones básicas |
| **Seaborn** | Gráficos estadísticos |
| **Scikit-learn** | Machine Learning |

#### Niveles de Análisis
1. **📊 Nivel 1: Pandas & NumPy**
   - Estadísticas descriptivas (.describe())
   - Agrupaciones (.groupby())
   - Operaciones vectorizadas
   - Matrices de correlación

2. **📈 Nivel 2: SciPy**
   - Test de normalidad (Shapiro-Wilk)
   - Test de Kruskal-Wallis
   - Intervalos de confianza
   - Test Chi-cuadrado
   - Correlación de Spearman

3. **🔬 Nivel 3: Scikit-Learn**
   - K-Means Clustering
   - PCA (Reducción de dimensionalidad)
   - Regresión Lineal
   - Detección de outliers
   - Preprocesamiento (StandardScaler, LabelEncoder)

#### Visualizaciones Generadas
1. `distribucion_precios.png` - Histograma con KDE
2. `precios_por_combustible.png` - Box plots comparativos
3. `matriz_correlacion.png` - Heatmap de correlaciones
4. `precio_vs_kilometraje.png` - Scatter plot con regresión
5. `ventas_por_region.png` - Gráfico de barras
6. `tendencia_temporal.png` - Serie temporal

#### Instalación
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

#### Ejecución
```bash
cd Analisis_02
python main.py
```

#### Ideal Para
- 🏢 Trabajar con herramientas profesionales
- ⚡ Análisis rápido de grandes datasets
- 📊 Crear visualizaciones de calidad
- 🤖 Aplicar machine learning

---

### 📁 Analisis_03: Análisis de Calidad de Agua

**🎯 Propósito:** Proyecto completo de análisis de calidad de agua con dos enfoques (manual y profesional).

#### Características
- 🐍 **3,188 líneas** de código total (2 sistemas completos)
- 💧 **1,106 muestras** de agua de Telangana, India (2018-2020)
- 📊 **2 sistemas:**
  - `main_basico.py` (1,476 líneas) - Implementación manual
  - `main_avanzado.py` (1,712 líneas) - Implementación profesional
- 🤖 **4 modelos de ML** (Random Forest: 95.2% accuracy)
- 📈 **Dashboards interactivos** con Plotly
- 🔬 **Análisis completo:** EDA, PCA, Clustering, ML

#### Parámetros Analizados
- **TDS** - Total Dissolved Solids (salinidad del agua)
- **SAR** - Sodium Absorption Ratio (peligro de sodio para riego)
- **RSC** - Residual Sodium Carbonate (aptitud para riego)
- **Clasificación** - C1S1 a C4S4, OG (calidad del agua)

#### Análisis Implementados
1. **📊 EDA** - Análisis exploratorio de datos
2. **🔍 Valores Faltantes** - Detección y manejo
3. **🧹 Preprocesamiento** - Limpieza y transformación
4. **📈 Correlación** - Relaciones entre variables
5. **🔬 PCA** - Reducción de dimensionalidad
6. **🎯 Clustering** - K-Means para segmentación
7. **🤖 Machine Learning** - 4 modelos predictivos:
   - Random Forest (95.2% accuracy)
   - Gradient Boosting
   - SVM
   - KNN

#### Ejecución
```bash
cd Analisis_03

# Sistema básico (manual)
python main_basico.py

# Sistema avanzado (profesional)
python main_avanzado.py
```

#### Ideal Para
- 🎓 Proyecto final integrador
- 🌍 Aplicación real (calidad de agua)
- 🤖 Práctica de machine learning
- 📊 Dashboards interactivos

---

## 🔄 Comparación de Proyectos

| Característica | Analisis_01 | Analisis_02 | Analisis_03 |
|----------------|-------------|-------------|-------------|
| **Líneas de código** | 2,157 | 2,100 | 3,188 (2 sistemas) |
| **Enfoque** | Manual | Profesional | Ambos |
| **Dependencias** | Ninguna | 6 librerías | 6 librerías |
| **Dataset** | BMW (50K) | BMW (50K) | Agua (1.1K) |
| **Visualizaciones** | No | 6 PNG | Dashboards HTML |
| **Machine Learning** | No | Sí (básico) | Sí (avanzado) |
| **Velocidad** | Lenta | Rápida | Rápida |
| **Dificultad** | Media | Media-Alta | Alta |
| **Tiempo estimado** | 1 semana | 1 semana | 1-2 semanas |

---

## 🎓 Ruta de Aprendizaje Recomendada

### Semana 1: Fundamentos del Análisis Manual
1. ✅ Estudia **Analisis_01/main.py**
2. ✅ Implementa las funciones estadísticas básicas
3. ✅ Completa ejercicios de nivel básico
4. ✅ Entiende cómo funcionan los algoritmos

### Semana 2: Análisis Profesional
1. ✅ Instala las librerías necesarias
2. ✅ Estudia **Analisis_02/main.py**
3. ✅ Compara con tu implementación manual
4. ✅ Experimenta con visualizaciones
5. ✅ Completa ejercicios de nivel intermedio

### Semana 3-4: Proyecto Final
1. ✅ Trabaja en **Analisis_03**
2. ✅ Comienza con `main_basico.py`
3. ✅ Avanza a `main_avanzado.py`
4. ✅ Implementa machine learning
5. ✅ Crea dashboards interactivos
6. ✅ Completa ejercicios de nivel avanzado

---

## 📊 Datasets Incluidos

### Dataset 1: Ventas de BMW (2010-2024)
- **Ubicación:** `Analisis_01/sample/` y `Analisis_02/sample/`
- **Archivo:** `BMW sales data (2010-2024) (1).csv`
- **Registros:** 50,000 ventas
- **Período:** 15 años (2010-2024)
- **Columnas:** 11 variables

**Columnas:**
- Model, Year, Region, Color, Fuel_Type
- Transmission, Engine_Size_L, Mileage_KM
- Price_USD, Sales_Volume, Sales_Classification

### Dataset 2: Calidad de Agua (2018-2020)
- **Ubicación:** `Analisis_03/samples/`
- **Registros:** 1,106 muestras
- **Período:** 3 años (2018-2020)
- **Origen:** Telangana, India
- **Parámetros:** TDS, SAR, RSC, Clasificación

---

## 🛠️ Requisitos

### Para Analisis_01 (Manual)
```bash
# Solo Python estándar
python --version  # Python 3.6+
```

### Para Analisis_02 y Analisis_03 (Profesional)
```bash
# Instalación rápida
pip install pandas numpy scipy matplotlib seaborn scikit-learn plotly

# O versiones específicas (recomendado)
pip install numpy==1.26.4 pandas==2.2.3 scipy==1.15.2 \
            matplotlib==3.9.2 seaborn==0.13.2 scikit-learn==1.5.1 \
            plotly==5.18.0
```

---

## 🚀 Inicio Rápido

### Opción 1: Comenzar con Análisis Manual
```bash
cd Clase_02/Analisis_01
python main.py
```

### Opción 2: Análisis Profesional
```bash
# Instalar dependencias
pip install pandas numpy scipy matplotlib seaborn scikit-learn

# Ejecutar
cd Clase_02/Analisis_02
python main.py
```

### Opción 3: Proyecto Completo de Calidad de Agua
```bash
cd Clase_02/Analisis_03

# Sistema básico
python main_basico.py

# Sistema avanzado
python main_avanzado.py
```

---

## 📖 Documentación Detallada

Cada proyecto tiene su propio README con información completa:

- **[Analisis_01/README.md](Analisis_01/README.md)** - Análisis manual (332 líneas)
- **[Analisis_02/README.md](Analisis_02/README.md)** - Análisis profesional (406 líneas)
- **[Analisis_03/README.md](Analisis_03/README.md)** - Calidad de agua (800 líneas)

---

## 🤝 Contribuciones

Este material es parte de un curso de Computación. Si encuentras errores o tienes sugerencias, por favor repórtalos.

---

## 📝 Licencia

Material educativo de uso libre para fines académicos.

---

## ✨ Autor

**David Marambio Salazar** - 2025

---

## 🎯 Objetivos de la Clase

Al completar Clase_02, serás capaz de:

- ✅ Implementar algoritmos estadísticos desde cero
- ✅ Usar Pandas para manipulación de datos
- ✅ Aplicar NumPy para computación numérica
- ✅ Realizar tests estadísticos con SciPy
- ✅ Crear visualizaciones con Matplotlib y Seaborn
- ✅ Implementar modelos de machine learning con Scikit-learn
- ✅ Desarrollar dashboards interactivos con Plotly
- ✅ Comparar enfoques manuales vs profesionales
- ✅ Completar proyectos de análisis de datos reales

---

**¡Comienza tu viaje en el análisis de datos! 📊🐍🚀**

