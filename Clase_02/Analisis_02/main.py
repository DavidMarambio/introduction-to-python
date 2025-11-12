"""
=============================================================================
ANALISIS 2: ANÁLISIS DE DATOS CON LIBRERÍAS PROFESIONALES
=============================================================================

Autor: Curso de Introducción a Programación
Descripción: Análisis de datos CSV usando librerías profesionales de Python
             (pandas, numpy, scipy, matplotlib, seaborn, scikit-learn)

Niveles:
  1. Estadística Básica con pandas y numpy
  2. Estadística Avanzada con scipy
  3. Ciencia de Datos con scikit-learn y visualizaciones

Librerías utilizadas:
  - pandas: Manipulación y análisis de datos
  - numpy: Operaciones numéricas y arrays
  - scipy: Estadística avanzada
  - matplotlib: Visualizaciones básicas
  - seaborn: Visualizaciones estadísticas avanzadas
  - scikit-learn: Machine learning y análisis predictivo

INSTALACIÓN DE DEPENDENCIAS:
----------------------------
Si encuentras errores de compatibilidad con NumPy 2.x, ejecuta:

  pip install numpy<2.0
  pip install --upgrade pandas scipy matplotlib seaborn scikit-learn

O instala todas las dependencias con versiones compatibles:

  pip install numpy==1.26.4 pandas==2.2.3 scipy==1.15.2 matplotlib==3.9.2 seaborn==0.13.2 scikit-learn==1.5.1

NOTA: Este script está diseñado para enseñar el uso de librerías profesionales
      en contraste con el enfoque manual del script 08_analisis_csv.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualizaciones
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =============================================================================
# FUNCIONES DE CARGA DE DATOS
# =============================================================================

def cargar_datos(archivo="./sample/BMW sales data (2010-2024) (1).csv"):
    """
    Carga datos CSV usando pandas.

    Args:
        archivo: Ruta del archivo CSV

    Returns:
        DataFrame de pandas con los datos
    """
    try:
        # Intentar diferentes encodings
        for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(archivo, encoding=encoding)
                print(f"✅ Archivo cargado exitosamente con encoding: {encoding}")
                print(f"📊 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
                return df
            except UnicodeDecodeError:
                continue
        
        # Si ningún encoding funciona
        df = pd.read_csv(archivo)
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{archivo}'")
        return None
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e}")
        return None


def mostrar_info_dataset(df):
    """Muestra información general del dataset."""
    print("\n" + "="*70)
    print("📋 INFORMACIÓN GENERAL DEL DATASET")
    print("="*70)
    
    print(f"\n📏 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    
    print("\n📊 Columnas y tipos de datos:")
    print(df.dtypes)
    
    print("\n🔍 Primeras 5 filas:")
    print(df.head())
    
    print("\n📈 Estadísticas descriptivas:")
    print(df.describe())
    
    print("\n❓ Valores nulos:")
    print(df.isnull().sum())
    
    print("\n💾 Uso de memoria:")
    print(f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


# =============================================================================
# NIVEL 1: ESTADÍSTICA BÁSICA CON PANDAS Y NUMPY
# =============================================================================

def estadistica_basica_pandas(df):
    """Análisis estadístico básico usando pandas y numpy."""
    print("\n" + "="*70)
    print("NIVEL 1: ESTADÍSTICA BÁSICA - Pandas & NumPy")
    print("="*70)

    print("\n📁 Archivo: BMW sales data (2010-2024) (1).csv")
    print("📋 Descripción: Análisis estadístico básico usando pandas para")
    print("               manipulación de datos y numpy para cálculos numéricos.")

    # 1. Análisis de frecuencias con value_counts()
    print("\n" + "─"*70)
    print("📊 ANÁLISIS DE FRECUENCIAS - value_counts()")
    print("─"*70)
    print("🎯 Objetivo: Identificar los modelos más populares en el dataset")
    print("🔧 Algoritmo: Cuenta la aparición de cada valor único en la columna")
    print("📈 Valores posibles: Enteros positivos (número de ocurrencias)")
    print("💡 Interpretación:")
    print("   • Valores altos = Modelos más vendidos/populares")
    print("   • Distribución uniforme = Ventas equilibradas entre modelos")
    print("   • Distribución desigual = Algunos modelos dominan el mercado")
    print()

    print("--- Top 10 Modelos Más Vendidos (usando value_counts) ---")
    top_modelos = df['Model'].value_counts().head(10)
    for modelo, cantidad in top_modelos.items():
        print(f"  {modelo}: {cantidad:,} ventas")

    print("\n✅ Interpretación de resultados:")
    max_ventas = top_modelos.max()
    min_ventas = top_modelos.min()
    diferencia = max_ventas - min_ventas
    print(f"   • Modelo más vendido: {top_modelos.index[0]} ({max_ventas:,} ventas)")
    print(f"   • Diferencia con el 10°: {diferencia:,} ventas ({diferencia/max_ventas*100:.1f}%)")
    if diferencia / max_ventas < 0.05:
        print("   • Conclusión: Ventas muy equilibradas entre modelos")
    else:
        print("   • Conclusión: Hay modelos claramente más populares")

    # 2. Estadísticas descriptivas con describe()
    print("\n" + "─"*70)
    print("📊 ESTADÍSTICAS DESCRIPTIVAS - describe()")
    print("─"*70)
    print("🎯 Objetivo: Obtener un resumen estadístico completo de los precios")
    print("🔧 Algoritmo: Calcula medidas de tendencia central y dispersión")
    print("📈 Valores posibles:")
    print("   • Promedio: Suma de valores / cantidad (sensible a outliers)")
    print("   • Mediana: Valor central (robusto a outliers)")
    print("   • Desv. Estándar: Dispersión promedio respecto a la media")
    print("💡 Interpretación:")
    print("   • Promedio ≈ Mediana → Distribución simétrica")
    print("   • Promedio > Mediana → Distribución sesgada a la derecha")
    print("   • Desv. Estándar alta → Precios muy variables")
    print()

    print("--- Estadísticas de Precios (USD) ---")
    precio_stats = df['Price_USD'].describe()
    promedio = precio_stats['mean']
    mediana = precio_stats['50%']
    desv_std = precio_stats['std']
    print(f"  Promedio: ${promedio:,.2f}")
    print(f"  Mediana (50%): ${mediana:,.2f}")
    print(f"  Desv. Estándar: ${desv_std:,.2f}")
    print(f"  Mínimo: ${precio_stats['min']:,.2f}")
    print(f"  Máximo: ${precio_stats['max']:,.2f}")

    print("\n✅ Interpretación de resultados:")
    diferencia_prom_med = abs(promedio - mediana)
    coef_variacion = (desv_std / promedio) * 100
    print(f"   • Diferencia Promedio-Mediana: ${diferencia_prom_med:,.2f}")
    if diferencia_prom_med / promedio < 0.01:
        print("   • Distribución: Aproximadamente simétrica")
    elif promedio > mediana:
        print("   • Distribución: Sesgada a la derecha (más valores altos)")
    else:
        print("   • Distribución: Sesgada a la izquierda (más valores bajos)")
    print(f"   • Coeficiente de Variación: {coef_variacion:.1f}%")
    if coef_variacion < 15:
        print("   • Variabilidad: Baja (precios homogéneos)")
    elif coef_variacion < 30:
        print("   • Variabilidad: Moderada")
    else:
        print("   • Variabilidad: Alta (precios muy dispersos)")
    
    # 3. Agrupaciones con groupby()
    print("\n--- Volumen Total de Ventas por Región (usando groupby) ---")
    ventas_region = df.groupby('Region')['Sales_Volume'].sum().sort_values(ascending=False)
    for region, volumen in ventas_region.items():
        print(f"  {region}: {volumen:,.0f} unidades")
    
    # 4. Operaciones con numpy
    print("\n--- Análisis con NumPy ---")
    precios_array = df['Price_USD'].values
    print(f"  Media (np.mean): ${np.mean(precios_array):,.2f}")
    print(f"  Mediana (np.median): ${np.median(precios_array):,.2f}")
    print(f"  Percentil 25: ${np.percentile(precios_array, 25):,.2f}")
    print(f"  Percentil 75: ${np.percentile(precios_array, 75):,.2f}")
    print(f"  Percentil 95: ${np.percentile(precios_array, 95):,.2f}")
    
    # 5. Filtrado avanzado con pandas
    print("\n--- Vehículos de Lujo (Precio > $100,000) ---")
    lujo = df[df['Price_USD'] > 100000]
    print(f"  Total: {len(lujo):,} vehículos ({len(lujo)/len(df)*100:.1f}%)")
    print(f"  Precio promedio: ${lujo['Price_USD'].mean():,.2f}")
    
    # 6. Crosstab para análisis cruzado
    print("\n--- Distribución: Tipo de Combustible × Transmisión ---")
    crosstab = pd.crosstab(df['Fuel_Type'], df['Transmission'])
    print(crosstab)

    # 7. Correlación básica
    print("\n" + "─"*70)
    print("📊 MATRIZ DE CORRELACIÓN - Pearson")
    print("─"*70)
    print("🎯 Objetivo: Medir la relación lineal entre variables numéricas")
    print("🔧 Algoritmo: Coeficiente de correlación de Pearson (r)")
    print("📈 Valores posibles: -1 a +1")
    print("   • r = +1: Correlación positiva perfecta")
    print("   • r = 0: Sin correlación lineal")
    print("   • r = -1: Correlación negativa perfecta")
    print("💡 Interpretación:")
    print("   • |r| > 0.7: Correlación fuerte")
    print("   • 0.3 < |r| < 0.7: Correlación moderada")
    print("   • |r| < 0.3: Correlación débil")
    print()

    print("--- Matriz de Correlación (Variables Numéricas) ---")
    columnas_numericas = ['Price_USD', 'Sales_Volume', 'Mileage_KM', 'Engine_Size_L']
    correlacion = df[columnas_numericas].corr()
    print(correlacion.round(4))

    print("\n✅ Interpretación de resultados:")
    # Encontrar las correlaciones más fuertes (excluyendo diagonal)
    corr_abs = correlacion.abs()
    np.fill_diagonal(corr_abs.values, 0)
    max_corr = corr_abs.max().max()
    if max_corr > 0.7:
        print(f"   • Correlación máxima: {max_corr:.4f} (FUERTE)")
        print("   • Hay variables con relación lineal fuerte")
    elif max_corr > 0.3:
        print(f"   • Correlación máxima: {max_corr:.4f} (MODERADA)")
        print("   • Hay variables con relación lineal moderada")
    else:
        print(f"   • Correlación máxima: {max_corr:.4f} (DÉBIL)")
        print("   • Las variables son mayormente independientes")
        print("   • No hay relaciones lineales fuertes entre variables")


# =============================================================================
# NIVEL 2: ESTADÍSTICA AVANZADA CON SCIPY
# =============================================================================

def estadistica_avanzada_scipy(df):
    """Análisis estadístico avanzado usando scipy."""
    print("\n" + "="*70)
    print("NIVEL 2: ESTADÍSTICA AVANZADA - SciPy")
    print("="*70)
    
    print("\n📁 Archivo: BMW sales data (2010-2024) (1).csv")
    print("📋 Descripción: Análisis estadístico avanzado usando scipy para")
    print("               pruebas de hipótesis, distribuciones y estadística inferencial.")
    
    # 1. Test de normalidad (Shapiro-Wilk)
    print("\n" + "─"*70)
    print("📊 TEST DE NORMALIDAD - Shapiro-Wilk")
    print("─"*70)
    print("🎯 Objetivo: Determinar si los datos siguen una distribución normal")
    print("🔧 Algoritmo: Compara la distribución observada con la normal teórica")
    print("📈 Valores posibles:")
    print("   • Estadístico W: 0 a 1 (1 = perfectamente normal)")
    print("   • P-valor: 0 a 1")
    print("💡 Interpretación:")
    print("   • p > 0.05: NO rechazamos normalidad (datos probablemente normales)")
    print("   • p ≤ 0.05: Rechazamos normalidad (datos NO normales)")
    print("   • Importante para decidir qué pruebas estadísticas usar")
    print()

    print("--- Test de Normalidad (Shapiro-Wilk) ---")
    muestra_precios = df['Price_USD'].sample(min(5000, len(df)), random_state=42)
    statistic, p_value = stats.shapiro(muestra_precios)
    print(f"  Estadístico W: {statistic:.6f}")
    print(f"  P-valor: {p_value:.6f}")
    if p_value > 0.05:
        print("  ✅ Los precios siguen una distribución normal (p > 0.05)")
    else:
        print("  ❌ Los precios NO siguen una distribución normal (p ≤ 0.05)")

    print("\n✅ Interpretación de resultados:")
    print(f"   • Estadístico W = {statistic:.6f}")
    if statistic > 0.99:
        print("   • Muy cercano a 1: Distribución casi perfectamente normal")
    elif statistic > 0.95:
        print("   • Cercano a 1: Distribución aproximadamente normal")
    else:
        print("   • Alejado de 1: Distribución claramente no normal")

    if p_value > 0.05:
        print("   • Recomendación: Usar pruebas paramétricas (t-test, ANOVA)")
    else:
        print("   • Recomendación: Usar pruebas no paramétricas (Mann-Whitney, Kruskal-Wallis)")

    # 2. Test de Kruskal-Wallis (comparación de múltiples grupos)
    print("\n" + "─"*70)
    print("📊 TEST DE KRUSKAL-WALLIS - Comparación de Grupos")
    print("─"*70)
    print("🎯 Objetivo: Comparar precios entre diferentes tipos de combustible")
    print("🔧 Algoritmo: Versión no paramétrica de ANOVA (no requiere normalidad)")
    print("   • Compara las medianas de 3+ grupos independientes")
    print("   • Basado en rangos, no en valores absolutos")
    print("📈 Valores posibles:")
    print("   • H-estadístico: ≥ 0 (valores altos = más diferencias)")
    print("   • P-valor: 0 a 1")
    print("💡 Interpretación:")
    print("   • p < 0.05: Al menos un grupo es diferente")
    print("   • p ≥ 0.05: No hay diferencias significativas entre grupos")
    print()

    print("--- Test de Kruskal-Wallis: Precios por Tipo de Combustible ---")
    grupos_combustible = [df[df['Fuel_Type'] == fuel]['Price_USD'].values
                          for fuel in df['Fuel_Type'].unique()]
    h_stat, p_value = stats.kruskal(*grupos_combustible)
    print(f"  H-estadístico: {h_stat:.4f}")
    print(f"  P-valor: {p_value:.6f}")
    print(f"  Grupos comparados: {len(grupos_combustible)} tipos de combustible")
    if p_value < 0.05:
        print("  ✅ Hay diferencias significativas entre grupos (p < 0.05)")
    else:
        print("  ❌ No hay diferencias significativas entre grupos (p ≥ 0.05)")

    print("\n✅ Interpretación de resultados:")
    if p_value < 0.05:
        print("   • Los precios varían significativamente según el combustible")
        print("   • Recomendación: Analizar qué tipo es más caro/barato")
    else:
        print("   • Los precios son similares entre tipos de combustible")
        print("   • El tipo de combustible NO afecta significativamente el precio")
    
    # 3. Intervalos de confianza
    print("\n--- Intervalo de Confianza 95% para Precio Promedio ---")
    precios = df['Price_USD'].values
    confidence_interval = stats.t.interval(
        confidence=0.95,
        df=len(precios)-1,
        loc=np.mean(precios),
        scale=stats.sem(precios)
    )
    print(f"  IC 95%: [${confidence_interval[0]:,.2f}, ${confidence_interval[1]:,.2f}]")
    print(f"  Media: ${np.mean(precios):,.2f}")
    
    # 4. Coeficiente de asimetría y curtosis
    print("\n--- Asimetría y Curtosis de Precios ---")
    skewness = stats.skew(precios)
    kurt = stats.kurtosis(precios)
    print(f"  Asimetría (Skewness): {skewness:.4f}")
    if abs(skewness) < 0.5:
        print("    → Distribución aproximadamente simétrica")
    elif skewness > 0:
        print("    → Distribución sesgada a la derecha")
    else:
        print("    → Distribución sesgada a la izquierda")
    
    print(f"  Curtosis: {kurt:.4f}")
    if abs(kurt) < 0.5:
        print("    → Distribución mesocúrtica (normal)")
    elif kurt > 0:
        print("    → Distribución leptocúrtica (colas pesadas)")
    else:
        print("    → Distribución platicúrtica (colas ligeras)")

    # 5. Test Chi-cuadrado de independencia
    print("\n" + "─"*70)
    print("📊 TEST CHI-CUADRADO - Independencia de Variables")
    print("─"*70)
    print("🎯 Objetivo: Determinar si dos variables categóricas están relacionadas")
    print("🔧 Algoritmo: Compara frecuencias observadas vs esperadas")
    print("   • H0: Las variables son independientes (no relacionadas)")
    print("   • H1: Las variables son dependientes (relacionadas)")
    print("📈 Valores posibles:")
    print("   • χ² (Chi-cuadrado): ≥ 0 (valores altos = más dependencia)")
    print("   • P-valor: 0 a 1")
    print("   • Grados de libertad: (filas-1) × (columnas-1)")
    print("💡 Interpretación:")
    print("   • p < 0.05: Rechazamos H0 → Variables DEPENDIENTES")
    print("   • p ≥ 0.05: No rechazamos H0 → Variables INDEPENDIENTES")
    print()

    print("--- Test Chi-cuadrado: Fuel_Type × Sales_Classification ---")
    contingency_table = pd.crosstab(df['Fuel_Type'], df['Sales_Classification'])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency_table)
    print(f"  Chi-cuadrado (χ²): {chi2:.4f}")
    print(f"  P-valor: {p_value:.6f}")
    print(f"  Grados de libertad: {dof}")
    if p_value < 0.05:
        print("  ✅ Las variables son dependientes (p < 0.05)")
    else:
        print("  ❌ Las variables son independientes (p ≥ 0.05)")

    print("\n✅ Interpretación de resultados:")
    if p_value < 0.05:
        print("   • El tipo de combustible SÍ influye en la clasificación de ventas")
        print("   • Hay una relación estadísticamente significativa")
        print("   • Recomendación: Analizar qué combustibles tienen mejores ventas")
    else:
        print("   • El tipo de combustible NO influye en la clasificación de ventas")
        print("   • Las variables son independientes")
        print("   • Las ventas altas/bajas ocurren por igual en todos los combustibles")
    
    # 6. Correlación de Spearman (no paramétrica)
    print("\n--- Correlación de Spearman: Precio vs Kilometraje ---")
    corr, p_value = stats.spearmanr(df['Price_USD'], df['Mileage_KM'])
    print(f"  Coeficiente de Spearman: {corr:.4f}")
    print(f"  P-valor: {p_value:.6f}")
    if abs(corr) > 0.7:
        print("  → Correlación fuerte")
    elif abs(corr) > 0.3:
        print("  → Correlación moderada")
    else:
        print("  → Correlación débil")
    
    # 7. Análisis de percentiles avanzado
    print("\n--- Análisis de Percentiles Detallado ---")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("  Percentiles de Precio:")
    for p in percentiles:
        valor = np.percentile(precios, p)
        print(f"    P{p}: ${valor:,.2f}")


# =============================================================================
# NIVEL 3: CIENCIA DE DATOS CON SCIKIT-LEARN
# =============================================================================

def ciencia_datos_sklearn(df):
    """Análisis de ciencia de datos usando scikit-learn."""
    print("\n" + "="*70)
    print("NIVEL 3: CIENCIA DE DATOS - Scikit-Learn & Machine Learning")
    print("="*70)
    
    print("\n📁 Archivo: BMW sales data (2010-2024) (1).csv")
    print("📋 Descripción: Análisis avanzado de ciencia de datos usando scikit-learn")
    print("               para clustering, PCA, regresión y análisis predictivo.")
    
    # Preparar datos
    df_ml = df.copy()
    
    # 1. Clustering con K-Means
    print("\n" + "─"*70)
    print("🤖 CLUSTERING K-MEANS - Segmentación Automática")
    print("─"*70)
    print("🎯 Objetivo: Agrupar vehículos similares automáticamente")
    print("🔧 Algoritmo: K-Means (aprendizaje no supervisado)")
    print("   1. Selecciona K centroides aleatorios")
    print("   2. Asigna cada punto al centroide más cercano")
    print("   3. Recalcula centroides como promedio del grupo")
    print("   4. Repite hasta convergencia")
    print("📈 Valores posibles:")
    print("   • Clusters: 0, 1, 2, ... (etiquetas de grupo)")
    print("   • Inercia: ≥ 0 (suma de distancias al centroide, menor = mejor)")
    print("💡 Interpretación:")
    print("   • Cada cluster representa un segmento de mercado")
    print("   • Inercia baja = clusters bien definidos")
    print("   • Distribución equilibrada = segmentos similares en tamaño")
    print()

    print("--- Clustering K-Means: Segmentación de Vehículos ---")
    features_clustering = ['Price_USD', 'Mileage_KM', 'Engine_Size_L', 'Sales_Volume']
    X_cluster = df_ml[features_clustering].copy()

    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_ml['Cluster'] = kmeans.fit_predict(X_scaled)

    print(f"  Número de clusters: 3")
    print(f"  Inercia: {kmeans.inertia_:.2f}")
    print("\n  Distribución de vehículos por cluster:")
    cluster_counts = df_ml['Cluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"    Cluster {cluster}: {count:,} vehículos ({count/len(df_ml)*100:.1f}%)")

    print("\n  Características promedio por cluster:")
    cluster_stats = df_ml.groupby('Cluster')[features_clustering].mean()
    print(cluster_stats.round(2))

    print("\n✅ Interpretación de resultados:")
    # Identificar características de cada cluster
    for cluster in range(3):
        stats = cluster_stats.loc[cluster]
        print(f"\n   Cluster {cluster}:")
        if stats['Price_USD'] < 60000:
            print(f"     • Segmento: ECONÓMICO (${stats['Price_USD']:,.0f})")
        elif stats['Price_USD'] < 90000:
            print(f"     • Segmento: MEDIO (${stats['Price_USD']:,.0f})")
        else:
            print(f"     • Segmento: PREMIUM (${stats['Price_USD']:,.0f})")
        print(f"     • Kilometraje promedio: {stats['Mileage_KM']:,.0f} km")
        print(f"     • Volumen de ventas: {stats['Sales_Volume']:,.0f} unidades")

    # 2. PCA - Análisis de Componentes Principales
    print("\n" + "─"*70)
    print("🤖 PCA - Análisis de Componentes Principales")
    print("─"*70)
    print("🎯 Objetivo: Reducir dimensiones manteniendo la información importante")
    print("🔧 Algoritmo: Principal Component Analysis")
    print("   • Encuentra direcciones de máxima varianza en los datos")
    print("   • Transforma datos a nuevas coordenadas (componentes principales)")
    print("   • Reduce de N dimensiones a 2-3 para visualización")
    print("📈 Valores posibles:")
    print("   • Varianza explicada: 0% a 100% por componente")
    print("   • Suma de varianzas: Información total retenida")
    print("💡 Interpretación:")
    print("   • >70% varianza total: Buena reducción dimensional")
    print("   • <50% varianza total: Se pierde mucha información")
    print("   • PC1 > PC2: Primera componente es más importante")
    print()

    print("--- PCA: Reducción de Dimensionalidad ---")
    pca = PCA(n_components=2)
    _ = pca.fit_transform(X_scaled)

    print(f"  Varianza explicada por componente:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {var*100:.2f}%")
    varianza_total = sum(pca.explained_variance_ratio_)*100
    print(f"  Varianza total explicada: {varianza_total:.2f}%")

    print("\n✅ Interpretación de resultados:")
    if varianza_total > 70:
        print(f"   • Excelente: {varianza_total:.1f}% de información retenida")
        print("   • Las 2 componentes capturan bien la estructura de los datos")
    elif varianza_total > 50:
        print(f"   • Aceptable: {varianza_total:.1f}% de información retenida")
        print("   • Se pierde algo de información pero es útil para visualización")
    else:
        print(f"   • Limitado: Solo {varianza_total:.1f}% de información retenida")
        print("   • Los datos tienen estructura compleja, difícil de reducir")

    # 3. Regresión Lineal: Predecir precio
    print("\n" + "─"*70)
    print("🤖 REGRESIÓN LINEAL - Predicción de Precios")
    print("─"*70)
    print("🎯 Objetivo: Predecir el precio basándose en características del vehículo")
    print("🔧 Algoritmo: Regresión Lineal (aprendizaje supervisado)")
    print("   • Encuentra la mejor línea/plano que ajusta los datos")
    print("   • Ecuación: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ")
    print("   • Minimiza el error cuadrático medio (MSE)")
    print("📈 Métricas de evaluación:")
    print("   • R² Score: -∞ a 1 (1 = predicción perfecta, 0 = modelo inútil)")
    print("   • RMSE: ≥ 0 (error promedio en unidades originales)")
    print("   • MAE: ≥ 0 (error absoluto promedio)")
    print("💡 Interpretación:")
    print("   • R² > 0.7: Modelo excelente")
    print("   • 0.3 < R² < 0.7: Modelo aceptable")
    print("   • R² < 0.3: Modelo pobre")
    print("   • Coeficientes positivos: Aumentan el precio")
    print("   • Coeficientes negativos: Disminuyen el precio")
    print()

    print("--- Regresión Lineal: Predicción de Precios ---")

    # Codificar variables categóricas
    le_fuel = LabelEncoder()
    le_trans = LabelEncoder()
    le_region = LabelEncoder()

    df_ml['Fuel_Type_Encoded'] = le_fuel.fit_transform(df_ml['Fuel_Type'])
    df_ml['Transmission_Encoded'] = le_trans.fit_transform(df_ml['Transmission'])
    df_ml['Region_Encoded'] = le_region.fit_transform(df_ml['Region'])

    # Features para regresión
    features_reg = ['Mileage_KM', 'Engine_Size_L', 'Sales_Volume',
                    'Fuel_Type_Encoded', 'Transmission_Encoded', 'Region_Encoded']
    X = df_ml[features_reg]
    y = df_ml['Price_USD']

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = np.mean(np.abs(y_test - y_pred))

    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE: ${mae:,.2f}")

    print("\n  Importancia de características (coeficientes):")
    for feature, coef in zip(features_reg, model.coef_):
        print(f"    {feature}: {coef:.2f}")

    print("\n✅ Interpretación de resultados:")
    if r2 > 0.7:
        print(f"   • R² = {r2:.4f}: EXCELENTE capacidad predictiva")
        print("   • El modelo explica >70% de la variabilidad en precios")
    elif r2 > 0.3:
        print(f"   • R² = {r2:.4f}: ACEPTABLE capacidad predictiva")
        print("   • El modelo captura algunas tendencias pero no todas")
    elif r2 > 0:
        print(f"   • R² = {r2:.4f}: POBRE capacidad predictiva")
        print("   • El modelo apenas mejora una predicción simple")
    else:
        print(f"   • R² = {r2:.4f}: MODELO INÚTIL")
        print("   • El modelo es peor que simplemente usar el promedio")

    print(f"   • Error promedio: ${mae:,.2f} (MAE)")
    print(f"   • Error cuadrático: ${rmse:,.2f} (RMSE)")

    # Identificar características más importantes
    coef_abs = [(feat, abs(coef)) for feat, coef in zip(features_reg, model.coef_)]
    coef_abs.sort(key=lambda x: x[1], reverse=True)
    print(f"   • Característica más influyente: {coef_abs[0][0]}")

    # 4. Análisis de outliers con IQR (método robusto)
    print("\n--- Detección de Outliers (Método IQR) ---")
    Q1 = df['Price_USD'].quantile(0.25)
    Q3 = df['Price_USD'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df['Price_USD'] < lower_bound) | (df['Price_USD'] > upper_bound)]
    print(f"  Outliers detectados: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"  Rango normal: ${lower_bound:,.2f} - ${upper_bound:,.2f}")

    # 5. Análisis de tendencias temporales
    print("\n--- Análisis de Tendencias Temporales ---")
    ventas_anuales = df.groupby('Year').agg({
        'Sales_Volume': 'sum',
        'Price_USD': 'mean'
    }).reset_index()

    print("  Evolución de ventas por año:")
    for _, row in ventas_anuales.head(5).iterrows():
        print(f"    {int(row['Year'])}: {row['Sales_Volume']:,.0f} unidades, Precio prom: ${row['Price_USD']:,.2f}")
    print("    ...")
    for _, row in ventas_anuales.tail(3).iterrows():
        print(f"    {int(row['Year'])}: {row['Sales_Volume']:,.0f} unidades, Precio prom: ${row['Price_USD']:,.2f}")

    # 6. Análisis de segmentación avanzada
    print("\n--- Segmentación Avanzada por Valor de Cliente ---")
    df_ml['Customer_Value'] = df_ml['Price_USD'] * df_ml['Sales_Volume']

    # Crear segmentos usando cuartiles
    df_ml['Value_Segment'] = pd.qcut(df_ml['Customer_Value'],
                                      q=4,
                                      labels=['Bajo', 'Medio', 'Alto', 'Premium'])

    segment_stats = df_ml.groupby('Value_Segment').agg({
        'Customer_Value': ['count', 'mean', 'sum']
    })

    print("  Distribución por segmento de valor:")
    for segment in ['Bajo', 'Medio', 'Alto', 'Premium']:
        count = segment_stats.loc[segment, ('Customer_Value', 'count')]
        mean_val = segment_stats.loc[segment, ('Customer_Value', 'mean')]
        total_val = segment_stats.loc[segment, ('Customer_Value', 'sum')]
        print(f"    {segment}: {count:,} vehículos, Valor prom: ${mean_val:,.2f}, Total: ${total_val:,.2f}")


# =============================================================================
# VISUALIZACIONES AVANZADAS - NIVEL EXPERTO
# =============================================================================

def crear_visualizaciones(df):
    """
    Crea visualizaciones profesionales de nivel experto en ciencia de datos.

    Incluye:
    - Análisis de distribuciones multivariadas
    - Visualizaciones estadísticas avanzadas
    - Dashboards interactivos
    - Análisis de segmentación
    - Mapas de calor avanzados
    - Análisis temporal sofisticado
    - Visualizaciones de machine learning
    """
    print("\n" + "="*70)
    print("🎨 VISUALIZACIONES PROFESIONALES - NIVEL EXPERTO")
    print("="*70)
    print("\n🎯 Objetivo: Crear visualizaciones de calidad publicable para análisis")
    print("📊 Total de visualizaciones: 15 gráficos profesionales")
    print("💾 Formato: PNG de alta resolución (300 DPI)")
    print("\n📁 Archivo: BMW sales data (2010-2024) (1).csv")
    print()

    # Configurar estilo profesional
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    sns.set_context("notebook", font_scale=1.2)

    # Colores profesionales
    colors_primary = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    colors_sequential = sns.color_palette("rocket", as_cmap=True)
    colors_diverging = sns.color_palette("vlag", as_cmap=True)

    # =========================================================================
    # 1. DASHBOARD MULTIVARIADO - Análisis Exploratorio Completo
    # =========================================================================
    print("\n📊 [1/15] Dashboard Exploratorio Multivariado")
    print("   → Análisis: Distribuciones, outliers, y estadísticas clave")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Subplot 1: Distribución de precios con estadísticas
    ax1 = fig.add_subplot(gs[0, :2])
    sns.histplot(data=df, x='Price_USD', kde=True, bins=60, color='#2E86AB', alpha=0.7, ax=ax1)
    mean_price = df['Price_USD'].mean()
    median_price = df['Price_USD'].median()
    ax1.axvline(mean_price, color='red', linestyle='--', linewidth=2, label=f'Media: ${mean_price:,.0f}')
    ax1.axvline(median_price, color='green', linestyle='--', linewidth=2, label=f'Mediana: ${median_price:,.0f}')
    ax1.set_title('Distribución de Precios con KDE', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Precio (USD)', fontsize=11)
    ax1.set_ylabel('Frecuencia', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Subplot 2: Estadísticas clave
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = f"""
    ESTADÍSTICAS CLAVE
    {'─'*25}

    Media:     ${mean_price:,.0f}
    Mediana:   ${median_price:,.0f}
    Desv. Std: ${df['Price_USD'].std():,.0f}

    Mínimo:    ${df['Price_USD'].min():,.0f}
    Máximo:    ${df['Price_USD'].max():,.0f}

    Q1 (25%):  ${df['Price_USD'].quantile(0.25):,.0f}
    Q3 (75%):  ${df['Price_USD'].quantile(0.75):,.0f}

    Registros: {len(df):,}
    """
    ax2.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Subplot 3: Boxplot por tipo de combustible
    ax3 = fig.add_subplot(gs[1, :])
    sns.boxplot(data=df, x='Fuel_Type', y='Price_USD', palette='Set2', ax=ax3)
    ax3.set_title('Distribución de Precios por Tipo de Combustible', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlabel('Tipo de Combustible', fontsize=11)
    ax3.set_ylabel('Precio (USD)', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)

    # Subplot 4: Violin plot por transmisión
    ax4 = fig.add_subplot(gs[2, 0])
    sns.violinplot(data=df, x='Transmission', y='Price_USD', palette='muted', ax=ax4)
    ax4.set_title('Distribución por Transmisión', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Transmisión', fontsize=10)
    ax4.set_ylabel('Precio (USD)', fontsize=10)
    ax4.tick_params(labelsize=9)

    # Subplot 5: Scatter plot Precio vs Kilometraje
    ax5 = fig.add_subplot(gs[2, 1])
    sample_data = df.sample(min(3000, len(df)))
    scatter = ax5.scatter(sample_data['Mileage_KM'], sample_data['Price_USD'],
                         c=sample_data['Engine_Size_L'], cmap='viridis', alpha=0.5, s=20)
    ax5.set_title('Precio vs Kilometraje', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Kilometraje (KM)', fontsize=10)
    ax5.set_ylabel('Precio (USD)', fontsize=10)
    plt.colorbar(scatter, ax=ax5, label='Tamaño Motor (L)')
    ax5.tick_params(labelsize=9)

    # Subplot 6: Count plot de modelos top
    ax6 = fig.add_subplot(gs[2, 2])
    top_models = df['Model'].value_counts().head(8)
    top_models.plot(kind='barh', ax=ax6, color='#F18F01')
    ax6.set_title('Top 8 Modelos', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Cantidad', fontsize=10)
    ax6.set_ylabel('Modelo', fontsize=10)
    ax6.tick_params(labelsize=9)

    fig.suptitle('DASHBOARD EXPLORATORIO - Análisis Multivariado BMW',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.savefig('01_dashboard_exploratorio.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =========================================================================
    # 2. MATRIZ DE CORRELACIÓN AVANZADA
    # =========================================================================
    print("📊 [2/15] Matriz de Correlación Avanzada con Clustering")
    print("   → Análisis: Correlaciones jerárquicas y agrupamiento")

    # Preparar datos numéricos
    numeric_cols = ['Price_USD', 'Sales_Volume', 'Mileage_KM', 'Engine_Size_L', 'Year']
    corr_matrix = df[numeric_cols].corr()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Heatmap con anotaciones
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={'label': 'Coeficiente de Correlación'},
                ax=ax1, vmin=-1, vmax=1)
    ax1.set_title('Matriz de Correlación de Pearson', fontsize=14, fontweight='bold', pad=15)

    # Clustermap (correlación con clustering jerárquico)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='vlag', center=0,
                square=True, linewidths=1, cbar_kws={'label': 'Correlación'},
                ax=ax2, vmin=-1, vmax=1)
    ax2.set_title('Matriz Triangular (sin duplicados)', fontsize=14, fontweight='bold', pad=15)

    fig.suptitle('ANÁLISIS DE CORRELACIONES - Variables Numéricas',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('02_matriz_correlacion_avanzada.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =========================================================================
    # 3. ANÁLISIS TEMPORAL AVANZADO
    # =========================================================================
    print("📊 [3/15] Análisis Temporal Multidimensional")
    print("   → Análisis: Tendencias, estacionalidad, y evolución de precios")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Evolución de ventas por año
    ventas_anuales = df.groupby('Year')['Sales_Volume'].sum().reset_index()
    ax1 = axes[0, 0]
    ax1.plot(ventas_anuales['Year'], ventas_anuales['Sales_Volume'],
             marker='o', linewidth=3, markersize=10, color='#2E86AB')
    ax1.fill_between(ventas_anuales['Year'], ventas_anuales['Sales_Volume'], alpha=0.3, color='#2E86AB')
    ax1.set_title('Evolución del Volumen de Ventas (2010-2024)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Año', fontsize=11)
    ax1.set_ylabel('Volumen de Ventas', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')

    # Evolución de precios promedio por año
    precios_anuales = df.groupby('Year')['Price_USD'].mean().reset_index()
    ax2 = axes[0, 1]
    ax2.plot(precios_anuales['Year'], precios_anuales['Price_USD'],
             marker='s', linewidth=3, markersize=10, color='#A23B72')
    ax2.fill_between(precios_anuales['Year'], precios_anuales['Price_USD'], alpha=0.3, color='#A23B72')
    ax2.set_title('Evolución del Precio Promedio (2010-2024)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Año', fontsize=11)
    ax2.set_ylabel('Precio Promedio (USD)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Heatmap: Ventas por Año y Tipo de Combustible
    ax3 = axes[1, 0]
    pivot_fuel = df.groupby(['Year', 'Fuel_Type'])['Sales_Volume'].sum().unstack(fill_value=0)
    sns.heatmap(pivot_fuel.T, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Ventas'})
    ax3.set_title('Ventas por Año y Tipo de Combustible', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Año', fontsize=11)
    ax3.set_ylabel('Tipo de Combustible', fontsize=11)

    # Distribución de ventas por región a lo largo del tiempo
    ax4 = axes[1, 1]
    ventas_region_year = df.groupby(['Year', 'Region'])['Sales_Volume'].sum().unstack()
    ventas_region_year.plot(kind='area', stacked=True, ax=ax4, alpha=0.7, colormap='tab10')
    ax4.set_title('Evolución de Ventas por Región (Stacked)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Año', fontsize=11)
    ax4.set_ylabel('Volumen de Ventas', fontsize=11)
    ax4.legend(title='Región', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('ANÁLISIS TEMPORAL - Tendencias y Evolución',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('03_analisis_temporal_avanzado.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =========================================================================
    # 4. ANÁLISIS DE SEGMENTACIÓN POR PRECIO
    # =========================================================================
    print("📊 [4/15] Segmentación de Mercado por Precio")
    print("   → Análisis: Segmentos de precio y características")

    # Crear segmentos de precio
    df_seg = df.copy()
    df_seg['Segmento_Precio'] = pd.cut(df_seg['Price_USD'],
                                        bins=[0, 50000, 75000, 100000, 150000],
                                        labels=['Económico', 'Medio', 'Premium', 'Lujo'])

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Distribución de segmentos
    ax1 = axes[0, 0]
    segment_counts = df_seg['Segmento_Precio'].value_counts()
    colors_seg = ['#6A994E', '#F18F01', '#A23B72', '#2E86AB']
    wedges, texts, autotexts = ax1.pie(segment_counts.values, labels=segment_counts.index,
                                        autopct='%1.1f%%', startangle=90, colors=colors_seg,
                                        textprops={'fontsize': 11, 'weight': 'bold'})
    ax1.set_title('Distribución de Vehículos por Segmento', fontsize=13, fontweight='bold')

    # Características por segmento
    ax2 = axes[0, 1]
    segment_stats = df_seg.groupby('Segmento_Precio').agg({
        'Price_USD': 'mean',
        'Sales_Volume': 'mean',
        'Mileage_KM': 'mean',
        'Engine_Size_L': 'mean'
    })
    segment_stats_norm = (segment_stats - segment_stats.min()) / (segment_stats.max() - segment_stats.min())
    segment_stats_norm.T.plot(kind='bar', ax=ax2, width=0.8, colormap='Set2')
    ax2.set_title('Características Normalizadas por Segmento', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Característica', fontsize=11)
    ax2.set_ylabel('Valor Normalizado (0-1)', fontsize=11)
    ax2.legend(title='Segmento', fontsize=9)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Boxplot comparativo
    ax3 = axes[1, 0]
    sns.boxplot(data=df_seg, x='Segmento_Precio', y='Sales_Volume', palette='Set3', ax=ax3)
    ax3.set_title('Volumen de Ventas por Segmento', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Segmento de Precio', fontsize=11)
    ax3.set_ylabel('Volumen de Ventas', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)

    # Heatmap: Segmento vs Tipo de Combustible
    ax4 = axes[1, 1]
    cross_tab = pd.crosstab(df_seg['Segmento_Precio'], df_seg['Fuel_Type'], normalize='index') * 100
    sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='Blues', ax=ax4, cbar_kws={'label': '% del Segmento'})
    ax4.set_title('Distribución de Combustible por Segmento (%)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Tipo de Combustible', fontsize=11)
    ax4.set_ylabel('Segmento de Precio', fontsize=11)

    fig.suptitle('SEGMENTACIÓN DE MERCADO - Análisis por Precio',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('04_segmentacion_mercado.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =========================================================================
    # 5. ANÁLISIS GEOGRÁFICO - VENTAS POR REGIÓN
    # =========================================================================
    print("📊 [5/15] Análisis Geográfico Detallado")
    print("   → Análisis: Ventas, precios y preferencias por región")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Ventas totales por región
    ax1 = axes[0, 0]
    ventas_region = df.groupby('Region')['Sales_Volume'].sum().sort_values(ascending=True)
    ventas_region.plot(kind='barh', ax=ax1, color=colors_primary)
    ax1.set_title('Volumen Total de Ventas por Región', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Volumen de Ventas', fontsize=11)
    ax1.set_ylabel('Región', fontsize=11)
    ax1.grid(axis='x', alpha=0.3)

    # Precio promedio por región
    ax2 = axes[0, 1]
    precio_region = df.groupby('Region')['Price_USD'].mean().sort_values(ascending=False)
    bars = ax2.bar(range(len(precio_region)), precio_region.values, color=colors_primary)
    ax2.set_xticks(range(len(precio_region)))
    ax2.set_xticklabels(precio_region.index, rotation=45, ha='right')
    ax2.set_title('Precio Promedio por Región', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Región', fontsize=11)
    ax2.set_ylabel('Precio Promedio (USD)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)

    # Preferencias de combustible por región
    ax3 = axes[1, 0]
    fuel_region = pd.crosstab(df['Region'], df['Fuel_Type'], normalize='index') * 100
    fuel_region.plot(kind='bar', stacked=True, ax=ax3, colormap='Set2')
    ax3.set_title('Preferencias de Combustible por Región (%)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Región', fontsize=11)
    ax3.set_ylabel('Porcentaje', fontsize=11)
    ax3.legend(title='Tipo de Combustible', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)

    # Scatter: Ventas vs Precio por región
    ax4 = axes[1, 1]
    region_summary = df.groupby('Region').agg({
        'Sales_Volume': 'sum',
        'Price_USD': 'mean'
    }).reset_index()

    scatter = ax4.scatter(region_summary['Sales_Volume'], region_summary['Price_USD'],
                         s=500, alpha=0.6, c=range(len(region_summary)), cmap='viridis')

    for idx, row in region_summary.iterrows():
        ax4.annotate(row['Region'], (row['Sales_Volume'], row['Price_USD']),
                    fontsize=10, ha='center', va='center', weight='bold')

    ax4.set_title('Ventas vs Precio Promedio por Región', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Volumen Total de Ventas', fontsize=11)
    ax4.set_ylabel('Precio Promedio (USD)', fontsize=11)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('ANÁLISIS GEOGRÁFICO - Ventas y Preferencias por Región',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('05_analisis_geografico.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("\n✅ Visualizaciones 1-5 completadas!")

    # =========================================================================
    # 6. ANÁLISIS DE MODELOS - TOP PERFORMERS
    # =========================================================================
    print("📊 [6/15] Análisis de Modelos Top Performers")
    print("   → Análisis: Modelos más vendidos y rentables")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Top 10 modelos por volumen de ventas
    ax1 = axes[0, 0]
    top_models_sales = df.groupby('Model')['Sales_Volume'].sum().sort_values(ascending=False).head(10)
    top_models_sales.plot(kind='barh', ax=ax1, color='#2E86AB')
    ax1.set_title('Top 10 Modelos por Volumen de Ventas', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Volumen Total de Ventas', fontsize=11)
    ax1.set_ylabel('Modelo', fontsize=11)
    ax1.grid(axis='x', alpha=0.3)

    # Top 10 modelos por precio promedio
    ax2 = axes[0, 1]
    top_models_price = df.groupby('Model')['Price_USD'].mean().sort_values(ascending=False).head(10)
    top_models_price.plot(kind='barh', ax=ax2, color='#A23B72')
    ax2.set_title('Top 10 Modelos por Precio Promedio', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Precio Promedio (USD)', fontsize=11)
    ax2.set_ylabel('Modelo', fontsize=11)
    ax2.grid(axis='x', alpha=0.3)

    # Distribución de modelos por tipo de combustible
    ax3 = axes[1, 0]
    model_fuel = pd.crosstab(df['Model'], df['Fuel_Type'])
    top_10_models = df['Model'].value_counts().head(10).index
    model_fuel_top = model_fuel.loc[top_10_models]
    model_fuel_top.plot(kind='bar', stacked=True, ax=ax3, colormap='Set3')
    ax3.set_title('Distribución de Combustible - Top 10 Modelos', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Modelo', fontsize=11)
    ax3.set_ylabel('Cantidad', fontsize=11)
    ax3.legend(title='Combustible', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)

    # Scatter: Precio vs Ventas por modelo (top 15)
    ax4 = axes[1, 1]
    model_summary = df.groupby('Model').agg({
        'Sales_Volume': 'sum',
        'Price_USD': 'mean'
    }).reset_index()
    top_15_models = df['Model'].value_counts().head(15).index
    model_summary_top = model_summary[model_summary['Model'].isin(top_15_models)]

    scatter = ax4.scatter(model_summary_top['Sales_Volume'], model_summary_top['Price_USD'],
                         s=300, alpha=0.6, c=range(len(model_summary_top)), cmap='plasma')

    for _, row in model_summary_top.iterrows():
        ax4.annotate(row['Model'], (row['Sales_Volume'], row['Price_USD']),
                    fontsize=8, ha='center', va='bottom')

    ax4.set_title('Precio vs Ventas - Top 15 Modelos', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Volumen Total de Ventas', fontsize=11)
    ax4.set_ylabel('Precio Promedio (USD)', fontsize=11)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('ANÁLISIS DE MODELOS - Top Performers',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('06_analisis_modelos.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =========================================================================
    # 7. ANÁLISIS DE DISTRIBUCIONES ESTADÍSTICAS
    # =========================================================================
    print("📊 [7/15] Análisis de Distribuciones Estadísticas")
    print("   → Análisis: QQ-plots, distribuciones y normalidad")

    from scipy import stats

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Histograma con curva normal
    ax1 = axes[0, 0]
    mu, sigma = df['Price_USD'].mean(), df['Price_USD'].std()
    n, bins, patches = ax1.hist(df['Price_USD'], bins=50, density=True, alpha=0.7, color='#2E86AB')
    y = stats.norm.pdf(bins, mu, sigma)
    ax1.plot(bins, y, 'r--', linewidth=2, label='Distribución Normal Teórica')
    ax1.set_title('Distribución de Precios vs Normal', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Precio (USD)', fontsize=10)
    ax1.set_ylabel('Densidad', fontsize=10)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # QQ-Plot para precios
    ax2 = axes[0, 1]
    stats.probplot(df['Price_USD'], dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot - Precios', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)

    # Boxplot comparativo múltiple
    ax3 = axes[0, 2]
    data_to_plot = [df[df['Fuel_Type'] == ft]['Price_USD'].values for ft in df['Fuel_Type'].unique()]
    bp = ax3.boxplot(data_to_plot, labels=df['Fuel_Type'].unique(), patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_primary):
        patch.set_facecolor(color)
    ax3.set_title('Boxplot Comparativo por Combustible', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Tipo de Combustible', fontsize=10)
    ax3.set_ylabel('Precio (USD)', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)

    # Distribución de kilometraje
    ax4 = axes[1, 0]
    sns.histplot(data=df, x='Mileage_KM', kde=True, bins=50, color='#F18F01', ax=ax4)
    ax4.set_title('Distribución de Kilometraje', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Kilometraje (KM)', fontsize=10)
    ax4.set_ylabel('Frecuencia', fontsize=10)
    ax4.grid(alpha=0.3)

    # Distribución de tamaño de motor
    ax5 = axes[1, 1]
    sns.histplot(data=df, x='Engine_Size_L', kde=True, bins=30, color='#6A994E', ax=ax5)
    ax5.set_title('Distribución de Tamaño de Motor', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Tamaño de Motor (L)', fontsize=10)
    ax5.set_ylabel('Frecuencia', fontsize=10)
    ax5.grid(alpha=0.3)

    # Distribución de volumen de ventas
    ax6 = axes[1, 2]
    sns.histplot(data=df, x='Sales_Volume', kde=True, bins=50, color='#C73E1D', ax=ax6)
    ax6.set_title('Distribución de Volumen de Ventas', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Volumen de Ventas', fontsize=10)
    ax6.set_ylabel('Frecuencia', fontsize=10)
    ax6.grid(alpha=0.3)

    fig.suptitle('ANÁLISIS DE DISTRIBUCIONES ESTADÍSTICAS',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('07_distribuciones_estadisticas.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("\n✅ Visualizaciones 6-7 completadas!")

    # =========================================================================
    # 8. PAIRPLOT - RELACIONES MULTIVARIADAS
    # =========================================================================
    print("📊 [8/15] Pairplot - Análisis de Relaciones Multivariadas")
    print("   → Análisis: Relaciones entre todas las variables numéricas")

    # Seleccionar muestra para pairplot (más rápido)
    sample_size = min(2000, len(df))
    df_sample = df.sample(sample_size, random_state=42)

    # Crear pairplot
    pairplot_vars = ['Price_USD', 'Mileage_KM', 'Engine_Size_L', 'Sales_Volume']
    g = sns.pairplot(df_sample[pairplot_vars + ['Fuel_Type']], hue='Fuel_Type',
                     palette='Set2', diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30},
                     height=3)
    g.fig.suptitle('PAIRPLOT - Relaciones Multivariadas (Muestra de 2,000 registros)',
                   fontsize=16, fontweight='bold', y=1.01)
    plt.savefig('08_pairplot_multivariado.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =========================================================================
    # 9. ANÁLISIS DE CLUSTERING (K-MEANS VISUALIZATION)
    # =========================================================================
    print("📊 [9/15] Visualización de Clustering K-Means")
    print("   → Análisis: Segmentación automática de vehículos")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Preparar datos para clustering
    features_clustering = ['Price_USD', 'Mileage_KM', 'Engine_Size_L', 'Sales_Volume']
    X_cluster = df[features_clustering].copy()

    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA para visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Scatter plot de clusters en espacio PCA
    ax1 = axes[0, 0]
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis',
                         alpha=0.6, s=20)
    ax1.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
               pca.transform(kmeans.cluster_centers_)[:, 1],
               c='red', marker='X', s=300, edgecolors='black', linewidths=2,
               label='Centroides')
    ax1.set_title('Clusters en Espacio PCA', fontsize=13, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)', fontsize=11)
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Cluster')

    # Distribución de clusters
    ax2 = axes[0, 1]
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    bars = ax2.bar(cluster_counts.index, cluster_counts.values, color=colors_primary)
    ax2.set_title('Distribución de Vehículos por Cluster', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Cluster', fontsize=11)
    ax2.set_ylabel('Cantidad de Vehículos', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # Añadir porcentajes
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}\n({height/len(clusters)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    # Características promedio por cluster
    ax3 = axes[1, 0]
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    cluster_means = df_clustered.groupby('Cluster')[features_clustering].mean()
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    cluster_means_norm.T.plot(kind='bar', ax=ax3, width=0.8, colormap='viridis')
    ax3.set_title('Características Normalizadas por Cluster', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Característica', fontsize=11)
    ax3.set_ylabel('Valor Normalizado (0-1)', fontsize=11)
    ax3.legend(title='Cluster', fontsize=9)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)

    # Heatmap de características por cluster
    ax4 = axes[1, 1]
    sns.heatmap(cluster_means.T, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4,
                cbar_kws={'label': 'Valor Promedio'})
    ax4.set_title('Heatmap de Características por Cluster', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Cluster', fontsize=11)
    ax4.set_ylabel('Característica', fontsize=11)

    fig.suptitle('ANÁLISIS DE CLUSTERING - K-Means con 4 Clusters',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('09_clustering_kmeans.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("\n✅ Visualizaciones 8-9 completadas!")

    # =========================================================================
    # 10. ANÁLISIS DE REGRESIÓN LINEAL
    # =========================================================================
    print("📊 [10/15] Análisis de Regresión Lineal")
    print("   → Análisis: Predicción de precios y residuos")

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    # Preparar datos
    X_reg = df[['Mileage_KM', 'Engine_Size_L', 'Sales_Volume']].copy()
    y_reg = df['Price_USD'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calcular métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    residuals = y_test - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Predicho vs Real
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=20, color='#2E86AB')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Predicción Perfecta')
    ax1.set_title(f'Predicho vs Real (R² = {r2:.4f})', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Precio Real (USD)', fontsize=11)
    ax1.set_ylabel('Precio Predicho (USD)', fontsize=11)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Distribución de residuos
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Residuo = 0')
    ax2.set_title('Distribución de Residuos', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Residuo (USD)', fontsize=11)
    ax2.set_ylabel('Frecuencia', fontsize=11)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Residuos vs Predicho
    ax3 = axes[1, 0]
    ax3.scatter(y_pred, residuals, alpha=0.5, s=20, color='#F18F01')
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_title('Residuos vs Valores Predichos', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Precio Predicho (USD)', fontsize=11)
    ax3.set_ylabel('Residuo (USD)', fontsize=11)
    ax3.grid(alpha=0.3)

    # Importancia de características
    ax4 = axes[1, 1]
    feature_importance = pd.DataFrame({
        'Feature': X_reg.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=True)

    colors_feat = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
    ax4.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors_feat)
    ax4.set_title('Coeficientes del Modelo de Regresión', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Coeficiente', fontsize=11)
    ax4.set_ylabel('Característica', fontsize=11)
    ax4.axvline(0, color='black', linestyle='-', linewidth=1)
    ax4.grid(axis='x', alpha=0.3)

    # Añadir métricas
    metrics_text = f'RMSE: ${rmse:,.2f}\nR²: {r2:.4f}'
    ax4.text(0.95, 0.95, metrics_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('ANÁLISIS DE REGRESIÓN LINEAL - Predicción de Precios',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('10_regresion_lineal.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =========================================================================
    # 11. DASHBOARD EJECUTIVO - KPIs
    # =========================================================================
    print("📊 [11/15] Dashboard Ejecutivo con KPIs")
    print("   → Análisis: Métricas clave del negocio")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)

    # KPI 1: Ventas Totales
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    total_sales = df['Sales_Volume'].sum()
    kpi_text = f"{total_sales:,.0f}"
    ax1.text(0.5, 0.6, kpi_text, fontsize=36, fontweight='bold',
            ha='center', va='center', color='#2E86AB')
    ax1.text(0.5, 0.3, 'VENTAS TOTALES', fontsize=14, ha='center', va='center')
    ax1.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='#2E86AB', linewidth=3))

    # KPI 2: Precio Promedio
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    avg_price = df['Price_USD'].mean()
    kpi_text = f"${avg_price:,.0f}"
    ax2.text(0.5, 0.6, kpi_text, fontsize=36, fontweight='bold',
            ha='center', va='center', color='#A23B72')
    ax2.text(0.5, 0.3, 'PRECIO PROMEDIO', fontsize=14, ha='center', va='center')
    ax2.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='#A23B72', linewidth=3))

    # KPI 3: Total de Modelos
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    total_models = df['Model'].nunique()
    kpi_text = f"{total_models}"
    ax3.text(0.5, 0.6, kpi_text, fontsize=36, fontweight='bold',
            ha='center', va='center', color='#F18F01')
    ax3.text(0.5, 0.3, 'MODELOS ÚNICOS', fontsize=14, ha='center', va='center')
    ax3.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='#F18F01', linewidth=3))

    # KPI 4: Total de Regiones
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    total_regions = df['Region'].nunique()
    kpi_text = f"{total_regions}"
    ax4.text(0.5, 0.6, kpi_text, fontsize=36, fontweight='bold',
            ha='center', va='center', color='#6A994E')
    ax4.text(0.5, 0.3, 'REGIONES', fontsize=14, ha='center', va='center')
    ax4.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='#6A994E', linewidth=3))

    # Gráfico 1: Tendencia de ventas
    ax5 = fig.add_subplot(gs[1, :2])
    ventas_year = df.groupby('Year')['Sales_Volume'].sum()
    ax5.plot(ventas_year.index, ventas_year.values, marker='o', linewidth=3,
            markersize=8, color='#2E86AB')
    ax5.fill_between(ventas_year.index, ventas_year.values, alpha=0.3, color='#2E86AB')
    ax5.set_title('Tendencia de Ventas Anuales', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Año', fontsize=11)
    ax5.set_ylabel('Ventas', fontsize=11)
    ax5.grid(alpha=0.3)

    # Gráfico 2: Top 5 regiones
    ax6 = fig.add_subplot(gs[1, 2:])
    top_regions = df.groupby('Region')['Sales_Volume'].sum().sort_values(ascending=False).head(5)
    top_regions.plot(kind='barh', ax=ax6, color=colors_primary)
    ax6.set_title('Top 5 Regiones por Ventas', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Ventas Totales', fontsize=11)
    ax6.grid(axis='x', alpha=0.3)

    # Gráfico 3: Distribución por combustible
    ax7 = fig.add_subplot(gs[2, :2])
    fuel_dist = df['Fuel_Type'].value_counts()
    ax7.pie(fuel_dist.values, labels=fuel_dist.index, autopct='%1.1f%%',
           colors=colors_primary, startangle=90)
    ax7.set_title('Distribución por Tipo de Combustible', fontsize=13, fontweight='bold')

    # Gráfico 4: Distribución por transmisión
    ax8 = fig.add_subplot(gs[2, 2:])
    trans_dist = df['Transmission'].value_counts()
    ax8.pie(trans_dist.values, labels=trans_dist.index, autopct='%1.1f%%',
           colors=['#2E86AB', '#A23B72'], startangle=90)
    ax8.set_title('Distribución por Tipo de Transmisión', fontsize=13, fontweight='bold')

    fig.suptitle('DASHBOARD EJECUTIVO - KPIs y Métricas Clave',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.savefig('11_dashboard_ejecutivo.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("\n✅ Visualizaciones 10-11 completadas!")

    # =========================================================================
    # 12. ANÁLISIS DE OUTLIERS Y ANOMALÍAS
    # =========================================================================
    print("📊 [12/15] Análisis de Outliers y Anomalías")
    print("   → Análisis: Detección de valores atípicos")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Boxplot con outliers marcados - Precio
    ax1 = axes[0, 0]
    bp1 = ax1.boxplot([df['Price_USD']], vert=True, patch_artist=True,
                      widths=0.5, showfliers=True)
    bp1['boxes'][0].set_facecolor('#2E86AB')
    bp1['boxes'][0].set_alpha(0.7)
    ax1.set_title('Detección de Outliers - Precio', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Precio (USD)', fontsize=11)
    ax1.set_xticklabels(['Precio'])
    ax1.grid(axis='y', alpha=0.3)

    # Añadir estadísticas
    Q1 = df['Price_USD'].quantile(0.25)
    Q3 = df['Price_USD'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = len(df[(df['Price_USD'] < lower_bound) | (df['Price_USD'] > upper_bound)])

    stats_text = f'Q1: ${Q1:,.0f}\nQ3: ${Q3:,.0f}\nIQR: ${IQR:,.0f}\nOutliers: {outliers_count}'
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Boxplot con outliers - Kilometraje
    ax2 = axes[0, 1]
    bp2 = ax2.boxplot([df['Mileage_KM']], vert=True, patch_artist=True,
                      widths=0.5, showfliers=True)
    bp2['boxes'][0].set_facecolor('#A23B72')
    bp2['boxes'][0].set_alpha(0.7)
    ax2.set_title('Detección de Outliers - Kilometraje', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Kilometraje (KM)', fontsize=11)
    ax2.set_xticklabels(['Kilometraje'])
    ax2.grid(axis='y', alpha=0.3)

    # Z-Score para detección de outliers
    ax3 = axes[1, 0]
    from scipy import stats as sp_stats
    z_scores = np.abs(sp_stats.zscore(df['Price_USD']))
    ax3.hist(z_scores, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
    ax3.axvline(3, color='red', linestyle='--', linewidth=2, label='Umbral Z=3')
    ax3.set_title('Distribución de Z-Scores - Precio', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Z-Score', fontsize=11)
    ax3.set_ylabel('Frecuencia', fontsize=11)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Scatter plot con outliers marcados
    ax4 = axes[1, 1]
    outlier_mask = (df['Price_USD'] < lower_bound) | (df['Price_USD'] > upper_bound)
    ax4.scatter(df[~outlier_mask]['Mileage_KM'], df[~outlier_mask]['Price_USD'],
               alpha=0.5, s=20, color='#2E86AB', label='Normal')
    ax4.scatter(df[outlier_mask]['Mileage_KM'], df[outlier_mask]['Price_USD'],
               alpha=0.8, s=50, color='red', marker='x', label='Outliers')
    ax4.set_title('Outliers en Precio vs Kilometraje', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Kilometraje (KM)', fontsize=11)
    ax4.set_ylabel('Precio (USD)', fontsize=11)
    ax4.legend()
    ax4.grid(alpha=0.3)

    fig.suptitle('ANÁLISIS DE OUTLIERS Y ANOMALÍAS',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('12_analisis_outliers.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =========================================================================
    # 13. ANÁLISIS DE COMPOSICIÓN Y PROPORCIONES
    # =========================================================================
    print("📊 [13/15] Análisis de Composición y Proporciones")
    print("   → Análisis: Distribuciones y proporciones del mercado")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Treemap simulado con barras apiladas - Modelos por región
    ax1 = axes[0, 0]
    top_5_models = df['Model'].value_counts().head(5).index
    model_region = df[df['Model'].isin(top_5_models)].groupby(['Model', 'Region']).size().unstack(fill_value=0)
    model_region.plot(kind='bar', stacked=True, ax=ax1, colormap='tab10')
    ax1.set_title('Top 5 Modelos por Región', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Modelo', fontsize=11)
    ax1.set_ylabel('Cantidad', fontsize=11)
    ax1.legend(title='Región', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Donut chart - Clasificación de ventas
    ax2 = axes[0, 1]
    sales_class = df['Sales_Classification'].value_counts()
    wedges, texts, autotexts = ax2.pie(sales_class.values, labels=sales_class.index,
                                        autopct='%1.1f%%', startangle=90,
                                        colors=colors_primary, pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax2.add_artist(centre_circle)
    ax2.set_title('Clasificación de Ventas', fontsize=13, fontweight='bold')

    # Stacked area - Evolución de combustibles
    ax3 = axes[0, 2]
    fuel_year = df.groupby(['Year', 'Fuel_Type']).size().unstack(fill_value=0)
    fuel_year.plot(kind='area', stacked=True, ax=ax3, alpha=0.7, colormap='Set2')
    ax3.set_title('Evolución de Tipos de Combustible', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Año', fontsize=11)
    ax3.set_ylabel('Cantidad', fontsize=11)
    ax3.legend(title='Combustible', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(alpha=0.3)

    # Waffle chart simulado - Distribución de colores
    ax4 = axes[1, 0]
    color_dist = df['Color'].value_counts().head(8)
    color_dist.plot(kind='barh', ax=ax4, color=colors_primary)
    ax4.set_title('Top 8 Colores Más Populares', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Cantidad', fontsize=11)
    ax4.set_ylabel('Color', fontsize=11)
    ax4.grid(axis='x', alpha=0.3)

    # Sunburst simulado - Jerarquía de ventas
    ax5 = axes[1, 1]
    region_fuel = df.groupby(['Region', 'Fuel_Type']).size().unstack(fill_value=0)
    region_fuel_pct = region_fuel.div(region_fuel.sum(axis=1), axis=0) * 100
    region_fuel_pct.plot(kind='bar', stacked=True, ax=ax5, colormap='Spectral')
    ax5.set_title('Distribución de Combustible por Región (%)', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Región', fontsize=11)
    ax5.set_ylabel('Porcentaje', fontsize=11)
    ax5.legend(title='Combustible', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    ax5.grid(axis='y', alpha=0.3)

    # Proporción de transmisiones por año
    ax6 = axes[1, 2]
    trans_year = pd.crosstab(df['Year'], df['Transmission'], normalize='index') * 100
    trans_year.plot(kind='area', stacked=True, ax=ax6, alpha=0.7, color=['#2E86AB', '#A23B72'])
    ax6.set_title('Evolución de Transmisiones (%)', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Año', fontsize=11)
    ax6.set_ylabel('Porcentaje', fontsize=11)
    ax6.legend(title='Transmisión', fontsize=9)
    ax6.grid(alpha=0.3)

    fig.suptitle('ANÁLISIS DE COMPOSICIÓN Y PROPORCIONES',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('13_composicion_proporciones.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("\n✅ Visualizaciones 12-13 completadas!")

    # =========================================================================
    # 14. ANÁLISIS COMPARATIVO MULTIDIMENSIONAL
    # =========================================================================
    print("📊 [14/15] Análisis Comparativo Multidimensional")
    print("   → Análisis: Comparaciones complejas entre múltiples variables")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Radar chart - Comparación de características por tipo de combustible
    ax1 = axes[0, 0]
    fuel_stats = df.groupby('Fuel_Type').agg({
        'Price_USD': 'mean',
        'Sales_Volume': 'mean',
        'Mileage_KM': 'mean',
        'Engine_Size_L': 'mean'
    })

    # Normalizar para radar chart
    fuel_stats_norm = (fuel_stats - fuel_stats.min()) / (fuel_stats.max() - fuel_stats.min())

    categories = list(fuel_stats_norm.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax1 = plt.subplot(2, 2, 1, projection='polar')
    for idx, fuel_type in enumerate(fuel_stats_norm.index):
        values = fuel_stats_norm.loc[fuel_type].values.tolist()
        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, label=fuel_type, color=colors_primary[idx])
        ax1.fill(angles, values, alpha=0.15, color=colors_primary[idx])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=9)
    ax1.set_ylim(0, 1)
    ax1.set_title('Radar Chart - Características por Combustible', fontsize=13,
                 fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax1.grid(True)

    # Heatmap de correlación por segmento
    ax2 = axes[0, 1]
    df_temp = df.copy()
    df_temp['Price_Segment'] = pd.qcut(df_temp['Price_USD'], q=3, labels=['Bajo', 'Medio', 'Alto'])

    segment_corr = df_temp.groupby('Price_Segment')[['Price_USD', 'Sales_Volume', 'Mileage_KM']].corr()
    segment_corr_price = segment_corr.xs('Price_USD', level=1)

    sns.heatmap(segment_corr_price, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                ax=ax2, cbar_kws={'label': 'Correlación'}, vmin=-1, vmax=1)
    ax2.set_title('Correlaciones por Segmento de Precio', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Variable', fontsize=11)
    ax2.set_ylabel('Segmento', fontsize=11)

    # Violin plot comparativo
    ax3 = axes[1, 0]
    df_sample_violin = df.sample(min(5000, len(df)))
    sns.violinplot(data=df_sample_violin, x='Fuel_Type', y='Price_USD',
                  hue='Transmission', split=True, palette='muted', ax=ax3)
    ax3.set_title('Distribución de Precios: Combustible × Transmisión', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Tipo de Combustible', fontsize=11)
    ax3.set_ylabel('Precio (USD)', fontsize=11)
    ax3.legend(title='Transmisión', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # Bubble chart - 3 dimensiones
    ax4 = axes[1, 1]
    model_summary = df.groupby('Model').agg({
        'Price_USD': 'mean',
        'Sales_Volume': 'sum',
        'Mileage_KM': 'mean'
    }).reset_index()

    top_15 = df['Model'].value_counts().head(15).index
    model_summary_top = model_summary[model_summary['Model'].isin(top_15)]

    scatter = ax4.scatter(model_summary_top['Price_USD'],
                         model_summary_top['Sales_Volume'],
                         s=model_summary_top['Mileage_KM']/500,  # Tamaño por kilometraje
                         c=range(len(model_summary_top)),
                         cmap='viridis', alpha=0.6, edgecolors='black', linewidth=1)

    ax4.set_title('Bubble Chart: Precio × Ventas × Kilometraje', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Precio Promedio (USD)', fontsize=11)
    ax4.set_ylabel('Ventas Totales', fontsize=11)
    ax4.grid(alpha=0.3)

    # Añadir leyenda de tamaño
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
    legend = ax4.legend(handles, ['Bajo KM', 'Medio-Bajo', 'Medio-Alto', 'Alto KM'],
                       loc="upper right", title="Kilometraje", fontsize=8)

    fig.suptitle('ANÁLISIS COMPARATIVO MULTIDIMENSIONAL',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('14_analisis_comparativo.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =========================================================================
    # 15. RESUMEN ESTADÍSTICO VISUAL COMPLETO
    # =========================================================================
    print("📊 [15/15] Resumen Estadístico Visual Completo")
    print("   → Análisis: Síntesis visual de todos los análisis")

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)

    # Panel 1: Distribución de precios con estadísticas
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    n, bins, patches = ax1.hist(df['Price_USD'], bins=60, color='#2E86AB', alpha=0.7, edgecolor='black')

    # Añadir líneas de percentiles
    percentiles = [25, 50, 75]
    colors_perc = ['green', 'orange', 'red']
    for p, c in zip(percentiles, colors_perc):
        val = df['Price_USD'].quantile(p/100)
        ax1.axvline(val, color=c, linestyle='--', linewidth=2, label=f'P{p}: ${val:,.0f}')

    ax1.set_title('DISTRIBUCIÓN DE PRECIOS CON PERCENTILES', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Precio (USD)', fontsize=11)
    ax1.set_ylabel('Frecuencia', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: Top 10 modelos
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    top_10_models = df['Model'].value_counts().head(10)
    bars = ax2.barh(range(len(top_10_models)), top_10_models.values, color=colors_primary)
    ax2.set_yticks(range(len(top_10_models)))
    ax2.set_yticklabels(top_10_models.index)
    ax2.set_title('TOP 10 MODELOS MÁS VENDIDOS', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cantidad de Ventas', fontsize=11)
    ax2.grid(axis='x', alpha=0.3)

    # Añadir valores
    for i, (bar, val) in enumerate(zip(bars, top_10_models.values)):
        ax2.text(val, i, f' {val:,}', va='center', fontsize=9)

    # Panel 3: Matriz de correlación compacta
    ax3 = fig.add_subplot(gs[2, 0:2])
    corr_vars = ['Price_USD', 'Sales_Volume', 'Mileage_KM', 'Engine_Size_L']
    corr_matrix = df[corr_vars].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                square=True, linewidths=2, cbar_kws={'label': 'Correlación'},
                ax=ax3, vmin=-1, vmax=1)
    ax3.set_title('MATRIZ DE CORRELACIÓN', fontsize=14, fontweight='bold')

    # Panel 4: Distribución por región
    ax4 = fig.add_subplot(gs[2, 2:4])
    region_sales = df.groupby('Region')['Sales_Volume'].sum().sort_values(ascending=True)
    region_sales.plot(kind='barh', ax=ax4, color=colors_primary)
    ax4.set_title('VENTAS POR REGIÓN', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Volumen de Ventas', fontsize=11)
    ax4.grid(axis='x', alpha=0.3)

    # Panel 5: Estadísticas clave en tabla
    ax5 = fig.add_subplot(gs[3, :2])
    ax5.axis('off')

    stats_data = [
        ['Métrica', 'Valor'],
        ['─'*30, '─'*30],
        ['Total de Registros', f'{len(df):,}'],
        ['Precio Promedio', f'${df["Price_USD"].mean():,.2f}'],
        ['Precio Mediano', f'${df["Price_USD"].median():,.2f}'],
        ['Desviación Estándar', f'${df["Price_USD"].std():,.2f}'],
        ['Ventas Totales', f'{df["Sales_Volume"].sum():,}'],
        ['Modelos Únicos', f'{df["Model"].nunique()}'],
        ['Regiones', f'{df["Region"].nunique()}'],
        ['Rango de Años', f'{df["Year"].min()}-{df["Year"].max()}'],
    ]

    table = ax5.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Estilo de la tabla
    for i in range(len(stats_data)):
        if i == 0:
            table[(i, 0)].set_facecolor('#2E86AB')
            table[(i, 1)].set_facecolor('#2E86AB')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        elif i % 2 == 0:
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 1)].set_facecolor('#E8E8E8')

    ax5.set_title('ESTADÍSTICAS CLAVE DEL DATASET', fontsize=14, fontweight='bold', pad=20)

    # Panel 6: Distribución de combustibles
    ax6 = fig.add_subplot(gs[3, 2:4])
    fuel_counts = df['Fuel_Type'].value_counts()
    wedges, texts, autotexts = ax6.pie(fuel_counts.values, labels=fuel_counts.index,
                                        autopct='%1.1f%%', startangle=90,
                                        colors=colors_primary,
                                        textprops={'fontsize': 10, 'weight': 'bold'})
    ax6.set_title('DISTRIBUCIÓN POR TIPO DE COMBUSTIBLE', fontsize=14, fontweight='bold')

    fig.suptitle('RESUMEN ESTADÍSTICO VISUAL COMPLETO - BMW Sales Data (2010-2024)',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.savefig('15_resumen_estadistico_completo.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("\n✅ Visualizaciones 14-15 completadas!")
    print("\n" + "="*70)
    print("🎉 ¡TODAS LAS VISUALIZACIONES HAN SIDO GENERADAS EXITOSAMENTE!")
    print("="*70)
    print("\n📁 Archivos generados (15 visualizaciones profesionales):")
    print("   01. 01_dashboard_exploratorio.png")
    print("   02. 02_matriz_correlacion_avanzada.png")
    print("   03. 03_analisis_temporal_avanzado.png")
    print("   04. 04_segmentacion_mercado.png")
    print("   05. 05_analisis_geografico.png")
    print("   06. 06_analisis_modelos.png")
    print("   07. 07_distribuciones_estadisticas.png")
    print("   08. 08_pairplot_multivariado.png")
    print("   09. 09_clustering_kmeans.png")
    print("   10. 10_regresion_lineal.png")
    print("   11. 11_dashboard_ejecutivo.png")
    print("   12. 12_analisis_outliers.png")
    print("   13. 13_composicion_proporciones.png")
    print("   14. 14_analisis_comparativo.png")
    print("   15. 15_resumen_estadistico_completo.png")
    print("\n💡 Todas las imágenes están en alta resolución (300 DPI)")
    print("📊 Listas para presentaciones, reportes y publicaciones")
    print("="*70)


# =============================================================================
# COMPARACIÓN: MANUAL VS PROFESIONAL
# =============================================================================

def comparacion_metodos(df):
    """Compara métodos manuales vs librerías profesionales."""
    print("\n" + "="*70)
    print("📊 COMPARACIÓN: Métodos Manuales vs Librerías Profesionales")
    print("="*70)

    import time

    # Calcular promedio
    print("\n--- Cálculo de Promedio ---")

    # Método manual
    start = time.time()
    suma = 0
    for precio in df['Price_USD']:
        suma += precio
    promedio_manual = suma / len(df)
    tiempo_manual = time.time() - start

    # Método pandas
    start = time.time()
    promedio_pandas = df['Price_USD'].mean()
    tiempo_pandas = time.time() - start

    # Método numpy
    start = time.time()
    promedio_numpy = np.mean(df['Price_USD'].values)
    tiempo_numpy = time.time() - start

    print(f"  Método Manual: ${promedio_manual:,.2f} (Tiempo: {tiempo_manual*1000:.4f} ms)")
    print(f"  Pandas .mean(): ${promedio_pandas:,.2f} (Tiempo: {tiempo_pandas*1000:.4f} ms)")
    print(f"  NumPy np.mean(): ${promedio_numpy:,.2f} (Tiempo: {tiempo_numpy*1000:.4f} ms)")
    print(f"  ⚡ Aceleración: {tiempo_manual/tiempo_numpy:.1f}x más rápido con NumPy")

    # Ventajas de librerías profesionales
    print("\n--- Ventajas de Librerías Profesionales ---")
    print("  ✅ Pandas:")
    print("     • Manejo eficiente de grandes datasets")
    print("     • Operaciones vectorizadas (más rápidas)")
    print("     • Funciones integradas para análisis")
    print("     • Manejo automático de valores nulos")
    print("     • Integración con otras librerías")

    print("\n  ✅ NumPy:")
    print("     • Operaciones matemáticas optimizadas")
    print("     • Arrays multidimensionales eficientes")
    print("     • Funciones estadísticas rápidas")
    print("     • Base para otras librerías científicas")

    print("\n  ✅ SciPy:")
    print("     • Pruebas estadísticas avanzadas")
    print("     • Distribuciones de probabilidad")
    print("     • Optimización y álgebra lineal")
    print("     • Métodos científicos validados")

    print("\n  ✅ Scikit-Learn:")
    print("     • Algoritmos de machine learning")
    print("     • Preprocesamiento de datos")
    print("     • Validación de modelos")
    print("     • Escalabilidad y rendimiento")

    print("\n  ✅ Matplotlib/Seaborn:")
    print("     • Visualizaciones profesionales")
    print("     • Gráficos estadísticos avanzados")
    print("     • Personalización completa")
    print("     • Exportación de alta calidad")


# =============================================================================
# EJERCICIOS PROPUESTOS
# =============================================================================

def mostrar_ejercicios():
    """Muestra ejercicios propuestos para practicar con librerías profesionales."""
    print("\n" + "="*70)
    print("📝 EJERCICIOS PROPUESTOS - LIBRERÍAS PROFESIONALES")
    print("="*70)

    print("\n📚 Objetivo: Aprender a usar pandas, numpy, scipy, matplotlib,")
    print("   seaborn y scikit-learn para análisis de datos profesional.")

    ejercicios = [
        {
            "nivel": "Básico - Pandas & NumPy",
            "emoji": "🟢",
            "ejercicios": [
                "1. Usa df.groupby() para calcular el precio promedio por modelo",
                "2. Filtra vehículos con df[df['Year'] > 2020] y calcula estadísticas",
                "3. Crea una nueva columna 'Price_per_KM' usando operaciones vectorizadas",
                "4. Usa pd.crosstab() para analizar Region × Fuel_Type",
                "5. Calcula percentiles con np.percentile() para diferentes columnas",
                "6. Usa df.sort_values() para ordenar por múltiples columnas",
                "7. Aplica df.pivot_table() para crear tabla resumen",
                "8. Usa np.where() para crear categorías de precio (bajo/medio/alto)"
            ]
        },
        {
            "nivel": "Intermedio - SciPy & Estadística",
            "emoji": "🟡",
            "ejercicios": [
                "1. Realiza un test t de Student para comparar precios entre dos regiones",
                "2. Calcula intervalos de confianza para diferentes variables",
                "3. Usa stats.pearsonr() para correlaciones con p-valores",
                "4. Aplica test de normalidad a diferentes columnas numéricas",
                "5. Realiza ANOVA para comparar precios entre múltiples grupos",
                "6. Calcula la distribución de probabilidad de los precios",
                "7. Usa stats.zscore() para detectar outliers",
                "8. Aplica transformaciones (log, sqrt) y evalúa normalidad"
            ]
        },
        {
            "nivel": "Avanzado - Machine Learning",
            "emoji": "🔴",
            "ejercicios": [
                "1. Entrena un modelo de regresión para predecir Sales_Volume",
                "2. Aplica K-Means con diferentes valores de k y evalúa con elbow method",
                "3. Usa PCA para reducir dimensionalidad y visualiza en 2D",
                "4. Implementa validación cruzada con cross_val_score",
                "5. Crea un pipeline de preprocesamiento + modelo",
                "6. Usa GridSearchCV para optimizar hiperparámetros",
                "7. Implementa Random Forest y compara con regresión lineal",
                "8. Aplica DBSCAN para clustering basado en densidad"
            ]
        },
        {
            "nivel": "Experto - Visualización & Análisis Completo",
            "emoji": "🏆",
            "ejercicios": [
                "1. Crea un dashboard completo con subplots (2x3 gráficos)",
                "2. Usa seaborn.pairplot() para visualizar relaciones múltiples",
                "3. Crea gráficos interactivos con plotly",
                "4. Implementa análisis de series temporales con tendencias",
                "5. Crea heatmaps animados para evolución temporal",
                "6. Diseña un informe automático con todas las métricas clave",
                "7. Implementa análisis de cohortes por año de venta",
                "8. Crea visualizaciones 3D con matplotlib"
            ]
        }
    ]

    for grupo in ejercicios:
        print(f"\n{grupo['emoji']} {grupo['nivel']}:")
        for ejercicio in grupo['ejercicios']:
            print(f"   {ejercicio}")

    print("\n" + "="*70)
    print("💡 Tips:")
    print("   • Consulta la documentación oficial de cada librería")
    print("   • Experimenta con diferentes parámetros")
    print("   • Compara resultados entre diferentes métodos")
    print("   • Visualiza siempre que sea posible")
    print("   • Valida tus resultados con múltiples enfoques")
    print("="*70)


# =============================================================================
# MENÚ PRINCIPAL
# =============================================================================

def mostrar_menu():
    """Muestra el menú principal."""
    print("\n" + "="*70)
    print("🎓 ACTIVIDAD 9: ANÁLISIS DE DATOS CON LIBRERÍAS PROFESIONALES")
    print("="*70)

    print("\nSelecciona un nivel de análisis:")
    print("\n  1️⃣  Nivel 1: Estadística Básica")
    print("      (Pandas & NumPy)")
    print("\n  2️⃣  Nivel 2: Estadística Avanzada")
    print("      (SciPy & Pruebas Estadísticas)")
    print("\n  3️⃣  Nivel 3: Ciencia de Datos")
    print("      (Scikit-Learn & Machine Learning)")
    print("\n  4️⃣  Bonus: Visualizaciones Profesionales")
    print("      (Matplotlib & Seaborn)")
    print("\n  5️⃣  Comparación: Manual vs Profesional")
    print("\n  6️⃣  Ver Ejercicios Propuestos")
    print("\n  7️⃣  Ejecutar Análisis Completo")
    print("\n  8️⃣  Información del Dataset")
    print("\n  0️⃣  Salir")

    print("\n" + "="*70)


def main():
    """Función principal del programa."""
    print("🌟" * 35)
    print("   ANÁLISIS DE DATOS CON LIBRERÍAS PROFESIONALES")
    print("🌟" * 35)

    print("\n📚 Librerías utilizadas:")
    print("   • pandas - Manipulación de datos")
    print("   • numpy - Computación numérica")
    print("   • scipy - Estadística avanzada")
    print("   • matplotlib - Visualizaciones")
    print("   • seaborn - Gráficos estadísticos")
    print("   • scikit-learn - Machine Learning")

    # Cargar datos
    print("\n📂 Cargando datos...")
    df = cargar_datos()

    if df is None:
        print("❌ No se pudo cargar el archivo. Programa terminado.")
        return

    while True:
        mostrar_menu()
        opcion = input("\n👉 Ingresa tu opción: ").strip()

        if opcion == "1":
            estadistica_basica_pandas(df)
            input("\n⏸️  Presiona Enter para volver al menú...")

        elif opcion == "2":
            estadistica_avanzada_scipy(df)
            input("\n⏸️  Presiona Enter para volver al menú...")

        elif opcion == "3":
            ciencia_datos_sklearn(df)
            input("\n⏸️  Presiona Enter para volver al menú...")

        elif opcion == "4":
            crear_visualizaciones(df)
            input("\n⏸️  Presiona Enter para volver al menú...")

        elif opcion == "5":
            comparacion_metodos(df)
            input("\n⏸️  Presiona Enter para volver al menú...")

        elif opcion == "6":
            mostrar_ejercicios()
            input("\n⏸️  Presiona Enter para volver al menú...")

        elif opcion == "7":
            print("\n🚀 Ejecutando análisis completo...\n")
            estadistica_basica_pandas(df)
            input("\n⏸️  Presiona Enter para continuar...")
            estadistica_avanzada_scipy(df)
            input("\n⏸️  Presiona Enter para continuar...")
            ciencia_datos_sklearn(df)
            input("\n⏸️  Presiona Enter para continuar...")
            crear_visualizaciones(df)
            input("\n⏸️  Presiona Enter para volver al menú...")

        elif opcion == "8":
            mostrar_info_dataset(df)
            input("\n⏸️  Presiona Enter para volver al menú...")

        elif opcion == "0":
            print("\n¡Hasta luego! 👋")
            print("💡 Recuerda: Las librerías profesionales hacen tu código más")
            print("   eficiente, legible y mantenible. ¡Sigue practicando!")
            break

        else:
            print("\n❌ Opción no válida. Por favor, selecciona una opción del menú.")


if __name__ == "__main__":
    main()

