# 🐍 Curso de Introducción a la Programación con Python

## 📚 Descripción General

Este repositorio contiene material educativo completo para un curso de introducción a la programación en Python, diseñado para estudiantes que están comenzando su camino en el desarrollo de software y análisis de datos.

El curso está dividido en **2 clases principales** que cubren desde los fundamentos de programación hasta análisis de datos profesional con machine learning.

---

## 📂 Estructura del Repositorio

```
clases/
│
├── 📁 Clase_01/                    # Fundamentos de Python
│   ├── 01_interaccion_basica.py   (67 líneas)
│   ├── 02_condicionales_if_else.py (90 líneas)
│   ├── 03_bucles_loops.py         (105 líneas)
│   ├── 04_listas.py               (120 líneas)
│   ├── 05_diccionarios.py         (120 líneas)
│   ├── 06_funciones.py            (219 líneas)
│   ├── 07_strings.py              (129 líneas)
│   ├── 08_archivos.py             (150 líneas)
│   ├── 09_excepciones.py          (180 líneas)
│   └── README.md
│
└── 📁 Clase_02/                    # Análisis de Datos
    │
    ├── 📁 Analisis_01/             # Análisis Manual (sin librerías)
    │   ├── main.py                 (2,157 líneas)
    │   ├── sample/
    │   │   └── BMW sales data (2010-2024) (1).csv
    │   └── README.md
    │
    ├── 📁 Analisis_02/             # Análisis Profesional (con librerías)
    │   ├── main.py                 (2,100 líneas)
    │   ├── sample/
    │   │   └── BMW sales data (2010-2024) (1).csv
    │   └── README.md
    │
    └── 📁 Analisis_03/             # Análisis de Calidad de Agua
        ├── main_basico.py          (1,476 líneas)
        ├── main_avanzado.py        (1,712 líneas)
        ├── samples/
        │   └── [1,106 muestras de agua]
        └── README.md
```

**Total:** 8,625 líneas de código | 12 scripts | 3 proyectos de análisis

---

## 🎯 Clase 01: Fundamentos de Python

### Descripción
Introducción a los conceptos fundamentales de programación en Python, desde la sintaxis básica hasta el manejo de excepciones.

### Contenido (9 Scripts)

| # | Script | Líneas | Temas Cubiertos |
|---|--------|--------|-----------------|
| 1 | `01_interaccion_basica.py` | 67 | print(), input(), conversión de tipos, operaciones matemáticas |
| 2 | `02_condicionales_if_else.py` | 90 | if, elif, else, operadores de comparación y lógicos |
| 3 | `03_bucles_loops.py` | 105 | for, while, range(), break, continue, bucles anidados |
| 4 | `04_listas.py` | 120 | Listas, métodos, slicing, list comprehension |
| 5 | `05_diccionarios.py` | 120 | Diccionarios, métodos, diccionarios anidados |
| 6 | `06_funciones.py` | 219 | Definición, parámetros, return, *args, **kwargs, lambda |
| 7 | `07_strings.py` | 129 | Métodos de strings, formateo, slicing, validaciones |
| 8 | `08_archivos.py` | 150 | Lectura/escritura, modos, with, CSV |
| 9 | `09_excepciones.py` | 180 | try-except, múltiples excepciones, raise, finally |

**Total Clase 01:** 1,180 líneas de código

### Objetivos de Aprendizaje
- ✅ Comprender la sintaxis básica de Python
- ✅ Dominar estructuras de control (if, for, while)
- ✅ Trabajar con estructuras de datos (listas, diccionarios)
- ✅ Crear y usar funciones
- ✅ Manipular cadenas de texto
- ✅ Leer y escribir archivos
- ✅ Manejar errores con excepciones

### Cómo Usar
```bash
cd Clase_01
python 01_interaccion_basica.py
```

Cada script es independiente y contiene:
- 📖 Explicaciones detalladas
- 💡 Ejemplos prácticos
- 🎯 Ejercicios propuestos

---

## 📊 Clase 02: Análisis de Datos

### Descripción
Introducción al análisis de datos en Python, desde implementaciones manuales hasta el uso de librerías profesionales y machine learning.

### Contenido (3 Proyectos)

---

### 📁 Analisis_01: Análisis Manual de Datos CSV

**Propósito:** Aprender cómo funcionan los algoritmos estadísticos implementándolos desde cero.

**Características:**
- 🐍 **2,157 líneas** de código Python puro
- ✅ **Sin dependencias** externas (solo módulos estándar)
- 📊 **25+ funciones** estadísticas implementadas manualmente
- 📈 **3 niveles** de análisis (Básico, Avanzado, Ciencia de Datos)
- 🚗 **50,000 registros** de ventas BMW (2010-2024)

**Niveles de Análisis:**
1. **Estadística Básica** - Promedios, sumas, mínimos, máximos, frecuencias
2. **Estadística Avanzada** - Mediana, moda, desviación estándar, percentiles, correlaciones
3. **Ciencia de Datos** - Outliers, ABC/Pareto, análisis temporal, segmentación

**Ejecución:**
```bash
cd Clase_02/Analisis_01
python main.py
```

**Ideal para:** Estudiantes que quieren entender los fundamentos antes de usar librerías.

---

### 📁 Analisis_02: Análisis Profesional con Librerías

**Propósito:** Aprender a usar las herramientas estándar de la industria para análisis de datos.

**Características:**
- 🐍 **2,100 líneas** de código profesional
- 📚 **6 librerías** profesionales (Pandas, NumPy, SciPy, Matplotlib, Seaborn, Scikit-learn)
- 🤖 **Machine Learning** implementado (K-Means, PCA, Regresión Lineal)
- 📊 **6 visualizaciones** PNG generadas automáticamente
- ⚡ **10-100x más rápido** que implementación manual
- 🚗 **50,000 registros** de ventas BMW (2010-2024)

**Librerías Utilizadas:**
- **Pandas** - Manipulación de datos con DataFrames
- **NumPy** - Computación numérica y arrays
- **SciPy** - Estadística avanzada y tests
- **Matplotlib** - Visualizaciones básicas
- **Seaborn** - Gráficos estadísticos
- **Scikit-learn** - Machine Learning

**Niveles de Análisis:**
1. **Pandas & NumPy** - Estadísticas descriptivas, agrupaciones, correlaciones
2. **SciPy** - Tests estadísticos (Shapiro-Wilk, Kruskal-Wallis, Chi-cuadrado)
3. **Scikit-Learn** - Clustering, PCA, Regresión, detección de outliers

**Instalación:**
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

**Ejecución:**
```bash
cd Clase_02/Analisis_02
python main.py
```

**Ideal para:** Estudiantes que quieren trabajar con herramientas profesionales.

---

### 📁 Analisis_03: Análisis de Calidad de Agua

**Propósito:** Proyecto completo de análisis de calidad de agua con dos enfoques (manual y profesional).

**Características:**
- 🐍 **3,188 líneas** de código total (2 sistemas)
- 💧 **1,106 muestras** de agua de Telangana, India (2018-2020)
- 📊 **2 sistemas completos:**
  - `main_basico.py` (1,476 líneas) - Implementación manual
  - `main_avanzado.py` (1,712 líneas) - Implementación profesional
- 🤖 **Machine Learning** con 4 modelos (Random Forest 95.2% accuracy)
- 📈 **Dashboards interactivos** con Plotly
- 🔬 **Análisis completo:** EDA, PCA, Clustering, ML

**Parámetros Analizados:**
- **TDS** - Total Dissolved Solids (salinidad)
- **SAR** - Sodium Absorption Ratio (peligro de sodio)
- **RSC** - Residual Sodium Carbonate (aptitud para riego)
- **Clasificación** - C1S1 a C4S4, OG (calidad del agua)

**Análisis Implementados:**
1. **EDA** - Análisis exploratorio de datos
2. **Valores Faltantes** - Detección y manejo
3. **Preprocesamiento** - Limpieza y transformación
4. **Correlación** - Relaciones entre variables
5. **PCA** - Reducción de dimensionalidad
6. **Clustering** - K-Means para segmentación
7. **Machine Learning** - 4 modelos predictivos

**Ejecución:**
```bash
cd Clase_02/Analisis_03

# Sistema básico (manual)
python main_basico.py

# Sistema avanzado (profesional)
python main_avanzado.py
```

**Ideal para:** Proyecto final que integra todos los conceptos aprendidos.

---

## 🎓 Ruta de Aprendizaje Recomendada

### Nivel 1: Fundamentos (Semanas 1-2)
1. ✅ Completa **Clase_01** en orden (scripts 01-09)
2. ✅ Practica los ejercicios de cada script
3. ✅ Crea tus propios programas simples

### Nivel 2: Análisis Manual (Semana 3)
1. ✅ Estudia **Analisis_01** para entender algoritmos
2. ✅ Implementa las funciones estadísticas
3. ✅ Completa los ejercicios propuestos

### Nivel 3: Análisis Profesional (Semana 4)
1. ✅ Aprende **Analisis_02** con librerías profesionales
2. ✅ Compara con tu implementación manual
3. ✅ Experimenta con visualizaciones

### Nivel 4: Proyecto Final (Semana 5)
1. ✅ Trabaja en **Analisis_03** (calidad de agua)
2. ✅ Implementa machine learning
3. ✅ Crea dashboards interactivos

---

## 📊 Estadísticas del Repositorio

| Métrica | Valor |
|---------|-------|
| **Total de líneas de código** | 8,625 |
| **Scripts en Clase_01** | 9 |
| **Proyectos en Clase_02** | 3 |
| **Funciones implementadas** | 50+ |
| **Datasets incluidos** | 2 |
| **Registros totales** | 51,106 |
| **Librerías profesionales** | 6 |
| **Modelos de ML** | 4 |
| **Visualizaciones** | 15+ |

---

## 🛠️ Requisitos

### Requisitos Mínimos (Clase_01 y Analisis_01)
- **Python 3.6+**
- Sin dependencias externas

### Requisitos Completos (Analisis_02 y Analisis_03)
- **Python 3.8+**
- **Librerías:**
  ```bash
  pip install pandas numpy scipy matplotlib seaborn scikit-learn plotly
  ```

### Instalación Recomendada
```bash
# Versiones específicas (compatibilidad garantizada)
pip install numpy==1.26.4 pandas==2.2.3 scipy==1.15.2 \
            matplotlib==3.9.2 seaborn==0.13.2 scikit-learn==1.5.1 \
            plotly==5.18.0
```

---

## 🚀 Inicio Rápido

### Opción 1: Comenzar desde Cero
```bash
# Clonar o descargar el repositorio
cd clases/Clase_01
python 01_interaccion_basica.py
```

### Opción 2: Ir Directo a Análisis de Datos
```bash
cd clases/Clase_02/Analisis_01
python main.py
```

### Opción 3: Análisis Profesional
```bash
# Instalar dependencias
pip install pandas numpy scipy matplotlib seaborn scikit-learn

# Ejecutar análisis
cd clases/Clase_02/Analisis_02
python main.py
```

---

## 📖 Documentación Adicional

Cada directorio contiene su propio README con información detallada:

- **[Clase_01/README.md](Clase_01/README.md)** - Fundamentos de Python
- **[Clase_02/Analisis_01/README.md](Clase_02/Analisis_01/README.md)** - Análisis manual
- **[Clase_02/Analisis_02/README.md](Clase_02/Analisis_02/README.md)** - Análisis profesional
- **[Clase_02/Analisis_03/README.md](Clase_02/Analisis_03/README.md)** - Calidad de agua

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

## 🎯 Objetivos del Curso

Al completar este curso, serás capaz de:

- ✅ Programar en Python con confianza
- ✅ Implementar algoritmos estadísticos desde cero
- ✅ Usar librerías profesionales (Pandas, NumPy, Scikit-learn)
- ✅ Realizar análisis exploratorio de datos (EDA)
- ✅ Crear visualizaciones profesionales
- ✅ Aplicar técnicas de machine learning
- ✅ Desarrollar proyectos completos de análisis de datos

---

**¡Comienza tu viaje en la programación y análisis de datos! 🚀🐍📊**

