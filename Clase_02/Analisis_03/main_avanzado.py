"""
Sistema Profesional de Análisis de Calidad de Agua
Autor: Experto en Ciencia de Datos Ambientales
Descripción: Análisis completo usando librerías profesionales de Python
             con explicaciones de algoritmos, interpretaciones y dashboards ejecutivos
Objetivo: Identificar las mejores fuentes de agua subterránea
"""

import warnings
warnings.filterwarnings('ignore')

# Librerías de análisis de datos
import pandas as pd
import numpy as np

# Librerías de visualización
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Librerías de estadística
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest, chi2_contingency

# Librerías de Machine Learning
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Utilidades
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json


# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

# Colores corporativos
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'excellent': '#00C851',
    'good': '#33b5e5',
    'fair': '#ffbb33',
    'poor': '#ff4444',
    'very_poor': '#CC0000'
}


# ============================================================================
# CLASE PRINCIPAL: ANALIZADOR DE CALIDAD DE AGUA
# ============================================================================

class AnalizadorCalidadAgua:
    """
    Sistema profesional de análisis de calidad de agua subterránea
    
    Características:
    - Análisis estadístico completo con Pandas y SciPy
    - Machine Learning con Scikit-learn
    - Visualizaciones interactivas con Plotly
    - Dashboards ejecutivos
    - Sistema de ranking de calidad
    """
    
    def __init__(self, directorio_datos: str = "samples"):
        """
        Inicializa el analizador
        
        Args:
            directorio_datos: Directorio con los archivos CSV
        """
        self.directorio_datos = directorio_datos
        self.df = None
        self.df_procesado = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.modelos = {}
        self.resultados = {}
        
        print("\n" + "="*80)
        print("🌊 SISTEMA PROFESIONAL DE ANÁLISIS DE CALIDAD DE AGUA SUBTERRÁNEA")
        print("="*80)
        print("\n📚 Librerías utilizadas:")
        print("   • Pandas & NumPy - Análisis de datos")
        print("   • Scikit-learn - Machine Learning")
        print("   • SciPy - Estadística avanzada")
        print("   • Plotly & Seaborn - Visualizaciones")
        print("\n🎯 Objetivo: Identificar las mejores fuentes de agua")
        print("="*80)
    
    def cargar_datos(self) -> pd.DataFrame:
        """
        Carga y combina todos los archivos CSV
        
        Returns:
            DataFrame con todos los datos combinados
        """
        print("\n📂 CARGANDO DATOS...")
        print("-" * 80)
        
        archivos = [f for f in os.listdir(self.directorio_datos) if f.endswith('.csv')]
        dfs = []
        
        for archivo in sorted(archivos):
            ruta = os.path.join(self.directorio_datos, archivo)
            df_temp = pd.read_csv(ruta)
            
            # Extraer año del nombre del archivo
            if '2018' in archivo:
                df_temp['año'] = 2018
            elif '2019' in archivo:
                df_temp['año'] = 2019
            elif '2020' in archivo:
                df_temp['año'] = 2020
            
            dfs.append(df_temp)
            print(f"   ✓ {archivo}: {len(df_temp)} registros")
        
        self.df = pd.concat(dfs, ignore_index=True)
        
        print(f"\n✅ Total de registros cargados: {len(self.df)}")
        print(f"✅ Columnas disponibles: {len(self.df.columns)}")
        print(f"✅ Período: {self.df['año'].min()} - {self.df['año'].max()}")
        print(f"✅ Distritos: {self.df['district'].nunique()}")
        
        return self.df
    
    def analisis_exploratorio(self):
        """
        Análisis Exploratorio de Datos (EDA)
        
        Algoritmo: Estadística Descriptiva
        - Calcula medidas de tendencia central (media, mediana)
        - Calcula medidas de dispersión (desviación estándar, IQR)
        - Identifica valores faltantes y outliers
        
        Interpretación:
        - Media: Valor promedio del parámetro
        - Mediana: Valor central, robusto a outliers
        - Desv. Est.: Variabilidad de los datos
        - CV%: Variabilidad relativa (>30% indica alta variabilidad)
        """
        print("\n" + "="*80)
        print("📊 ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("="*80)
        
        print("\n🔍 INFORMACIÓN GENERAL DEL DATASET")
        print("-" * 80)
        print(f"Dimensiones: {self.df.shape[0]} filas × {self.df.shape[1]} columnas")
        print(f"\nTipos de datos:")
        print(self.df.dtypes.value_counts())
        
        print("\n📈 ESTADÍSTICAS DESCRIPTIVAS")
        print("-" * 80)
        
        # Seleccionar columnas numéricas
        columnas_numericas = self.df.select_dtypes(include=[np.number]).columns
        columnas_analisis = [col for col in columnas_numericas 
                            if col not in ['sno', 'año', 'lat_gis', 'long_gis']]
        
        # Estadísticas descriptivas
        stats_df = self.df[columnas_analisis].describe().T
        stats_df['cv%'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)
        stats_df['missing'] = self.df[columnas_analisis].isnull().sum()
        stats_df['missing%'] = (stats_df['missing'] / len(self.df) * 100).round(2)
        
        print("\nParámetros principales:")
        print(stats_df[['mean', 'std', 'cv%', 'min', '50%', 'max', 'missing%']].round(2))
        
        print("\n💡 INTERPRETACIÓN:")
        print("-" * 80)
        
        # Identificar parámetros con alta variabilidad
        alta_variabilidad = stats_df[stats_df['cv%'] > 50].index.tolist()
        if alta_variabilidad:
            print(f"⚠️  Parámetros con alta variabilidad (CV > 50%):")
            for param in alta_variabilidad[:5]:
                cv = stats_df.loc[param, 'cv%']
                print(f"   • {param}: CV = {cv:.1f}% (alta heterogeneidad espacial)")
        
        # Identificar valores faltantes
        missing_cols = stats_df[stats_df['missing%'] > 0].index.tolist()
        if missing_cols:
            print(f"\n⚠️  Columnas con valores faltantes:")
            for col in missing_cols[:5]:
                pct = stats_df.loc[col, 'missing%']
                print(f"   • {col}: {pct:.1f}% faltante")
        
        print("\n✅ CONCLUSIÓN:")
        print("   Los datos muestran variabilidad natural esperada en aguas subterráneas.")
        print("   Se requiere análisis detallado para identificar patrones espaciales.")
        
        return stats_df
    
    def analisis_valores_faltantes(self):
        """
        Análisis de valores faltantes
        
        Algoritmo: Análisis de Missing Data
        - Identifica patrones de datos faltantes
        - Calcula porcentajes por columna
        - Sugiere estrategias de imputación
        """
        print("\n" + "="*80)
        print("🔍 ANÁLISIS DE VALORES FALTANTES")
        print("="*80)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Faltantes': missing,
            'Porcentaje': missing_pct
        })
        missing_df = missing_df[missing_df['Faltantes'] > 0].sort_values('Faltantes', ascending=False)
        
        if len(missing_df) > 0:
            print("\n📊 Columnas con valores faltantes:")
            print(missing_df)
            
            print("\n💡 ESTRATEGIA DE TRATAMIENTO:")
            print("-" * 80)
            print("   • < 5% faltante: Imputación con mediana (robusto a outliers)")
            print("   • 5-20% faltante: Imputación con KNN o regresión")
            print("   • > 20% faltante: Considerar eliminar columna o análisis separado")
        else:
            print("\n✅ No se encontraron valores faltantes en el dataset")
        
        return missing_df

    def preprocesar_datos(self):
        """
        Preprocesamiento de datos

        Algoritmo: Data Cleaning & Feature Engineering
        - Imputación de valores faltantes
        - Detección y tratamiento de outliers
        - Normalización de variables
        - Creación de features derivadas

        Interpretación:
        - Outliers: Valores > Q3 + 1.5*IQR o < Q1 - 1.5*IQR
        - Normalización: Escala [0,1] para comparabilidad
        """
        print("\n" + "="*80)
        print("🔧 PREPROCESAMIENTO DE DATOS")
        print("="*80)

        self.df_procesado = self.df.copy()

        # 0. Convertir columnas numéricas clave a tipo numérico
        print("\n0️⃣  Convirtiendo columnas a tipo numérico...")
        columnas_clave = ['pH', 'TDS', 'E.C', 'T.H', 'SAR', 'RSC', 'Ca', 'Mg', 'Na', 'K', 'Cl',
                         'HCO3', 'CO3', 'SO4', 'F', 'NO3', 'gwl']

        for col in columnas_clave:
            if col in self.df_procesado.columns:
                self.df_procesado[col] = pd.to_numeric(self.df_procesado[col], errors='coerce')

        # 1. Imputación de valores faltantes
        print("\n1️⃣  Imputación de valores faltantes...")
        columnas_numericas = self.df_procesado.select_dtypes(include=[np.number]).columns

        for col in columnas_numericas:
            if self.df_procesado[col].isnull().sum() > 0:
                mediana = self.df_procesado[col].median()
                self.df_procesado[col].fillna(mediana, inplace=True)
                print(f"   ✓ {col}: Imputado con mediana = {mediana:.2f}")

        # 2. Detección de outliers
        print("\n2️⃣  Detección de outliers (método IQR)...")
        outliers_count = {}

        parametros_clave = ['pH', 'TDS', 'E.C', 'T.H', 'SAR', 'RSC']
        for param in parametros_clave:
            if param in self.df_procesado.columns:
                try:
                    # Asegurar que la columna sea numérica
                    col_numeric = pd.to_numeric(self.df_procesado[param], errors='coerce')

                    # Calcular cuantiles solo si hay datos válidos
                    if col_numeric.notna().sum() > 0:
                        Q1 = col_numeric.quantile(0.25)
                        Q3 = col_numeric.quantile(0.75)
                        IQR = Q3 - Q1

                        if IQR > 0:
                            outliers = ((col_numeric < (Q1 - 1.5 * IQR)) |
                                       (col_numeric > (Q3 + 1.5 * IQR))).sum()
                            outliers_count[param] = outliers
                            if outliers > 0:
                                pct = (outliers / len(self.df_procesado) * 100)
                                print(f"   ⚠️  {param}: {outliers} outliers ({pct:.1f}%)")
                except Exception:
                    # Si hay error, saltar esta columna
                    print(f"   ⚠️  {param}: No se pudo analizar (datos no numéricos)")
                    continue

        # 3. Crear features derivadas
        print("\n3️⃣  Creación de features derivadas...")

        # Índice de calidad general (basado en TDS y SAR)
        if 'TDS' in self.df_procesado.columns and 'SAR' in self.df_procesado.columns:
            # Normalizar TDS y SAR
            tds_norm = (self.df_procesado['TDS'] - self.df_procesado['TDS'].min()) / \
                       (self.df_procesado['TDS'].max() - self.df_procesado['TDS'].min())
            sar_norm = (self.df_procesado['SAR'] - self.df_procesado['SAR'].min()) / \
                       (self.df_procesado['SAR'].max() - self.df_procesado['SAR'].min())

            # Índice de calidad (menor es mejor)
            self.df_procesado['quality_index'] = (tds_norm * 0.6 + sar_norm * 0.4)
            print("   ✓ quality_index: Índice compuesto de calidad (0=mejor, 1=peor)")

        # Ratio Ca/Mg (importante para dureza)
        if 'Ca' in self.df_procesado.columns and 'Mg' in self.df_procesado.columns:
            self.df_procesado['Ca_Mg_ratio'] = self.df_procesado['Ca'] / (self.df_procesado['Mg'] + 0.001)
            print("   ✓ Ca_Mg_ratio: Relación Calcio/Magnesio")

        # Clasificación de pH
        if 'pH' in self.df_procesado.columns:
            try:
                # Asegurar que pH sea numérico
                ph_numeric = pd.to_numeric(self.df_procesado['pH'], errors='coerce')
                self.df_procesado['pH_category'] = pd.cut(
                    ph_numeric,
                    bins=[0, 6.5, 7.5, 8.5, 14],
                    labels=['Ácido', 'Neutro', 'Ligeramente Alcalino', 'Alcalino']
                )
                print("   ✓ pH_category: Clasificación de pH")
            except Exception:
                print("   ⚠️  pH_category: No se pudo crear (datos no numéricos)")

        print("\n✅ Preprocesamiento completado")
        print(f"   Dataset procesado: {self.df_procesado.shape[0]} filas × {self.df_procesado.shape[1]} columnas")

        return self.df_procesado

    def analisis_correlacion(self):
        """
        Análisis de Correlación

        Algoritmo: Correlación de Pearson
        - Mide relación lineal entre variables (-1 a +1)
        - r > 0.7: Correlación fuerte positiva
        - r < -0.7: Correlación fuerte negativa
        - |r| < 0.3: Correlación débil

        Interpretación:
        - Correlaciones altas indican redundancia de información
        - Útil para selección de features en ML
        - Identifica relaciones físico-químicas
        """
        print("\n" + "="*80)
        print("📊 ANÁLISIS DE CORRELACIÓN")
        print("="*80)

        # Seleccionar parámetros clave
        parametros = ['pH', 'E.C', 'TDS', 'T.H', 'SAR', 'RSC', 'Ca', 'Mg', 'Na', 'Cl']
        parametros_disponibles = [p for p in parametros if p in self.df_procesado.columns]

        # Calcular matriz de correlación
        corr_matrix = self.df_procesado[parametros_disponibles].corr()

        print("\n📈 Matriz de Correlación (Pearson):")
        print("-" * 80)
        print(corr_matrix.round(3))

        # Identificar correlaciones fuertes
        print("\n💡 CORRELACIONES FUERTES (|r| > 0.7):")
        print("-" * 80)

        correlaciones_fuertes = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    param1 = corr_matrix.columns[i]
                    param2 = corr_matrix.columns[j]
                    r = corr_matrix.iloc[i, j]
                    correlaciones_fuertes.append((param1, param2, r))

                    if r > 0:
                        print(f"   ✓ {param1} ↔ {param2}: r = {r:.3f} (fuerte positiva)")
                        print(f"      → Cuando {param1} aumenta, {param2} también aumenta")
                    else:
                        print(f"   ✓ {param1} ↔ {param2}: r = {r:.3f} (fuerte negativa)")
                        print(f"      → Cuando {param1} aumenta, {param2} disminuye")

        if not correlaciones_fuertes:
            print("   No se encontraron correlaciones fuertes (|r| > 0.7)")

        print("\n✅ CONCLUSIÓN:")
        print("   Las correlaciones identificadas reflejan relaciones físico-químicas naturales.")
        print("   TDS-EC alta es esperada (conductividad depende de sólidos disueltos).")

        # Crear visualización
        self._plot_correlation_heatmap(corr_matrix)

        return corr_matrix

    def _plot_correlation_heatmap(self, corr_matrix):
        """Crea heatmap de correlación"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlación")
        ))

        fig.update_layout(
            title="Matriz de Correlación - Parámetros de Calidad de Agua",
            xaxis_title="Parámetros",
            yaxis_title="Parámetros",
            width=900,
            height=800
        )

        fig.write_html("output_correlacion_heatmap.html")
        print("\n📊 Visualización guardada: output_correlacion_heatmap.html")

    def analisis_pca(self, n_components=5):
        """
        Análisis de Componentes Principales (PCA)

        Algoritmo: Principal Component Analysis
        - Reduce dimensionalidad preservando varianza
        - Transforma variables correlacionadas en componentes independientes
        - Cada componente es combinación lineal de variables originales

        Matemática:
        - PC1 captura máxima varianza
        - PC2 captura máxima varianza restante (ortogonal a PC1)
        - Continúa hasta n componentes

        Interpretación:
        - Varianza explicada: % de información capturada
        - Loadings: Contribución de cada variable al componente
        - Scores: Posición de cada muestra en espacio reducido

        Aplicación:
        - Identificar patrones principales en calidad de agua
        - Reducir redundancia de información
        - Visualizar similitudes entre muestras
        """
        print("\n" + "="*80)
        print("🔬 ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)")
        print("="*80)

        print("\n📚 ALGORITMO: Principal Component Analysis")
        print("-" * 80)
        print("PCA transforma variables correlacionadas en componentes independientes")
        print("que capturan la máxima varianza de los datos.")
        print("\nPasos del algoritmo:")
        print("1. Estandarizar datos (media=0, std=1)")
        print("2. Calcular matriz de covarianza")
        print("3. Obtener eigenvectores y eigenvalores")
        print("4. Ordenar por eigenvalores (varianza explicada)")
        print("5. Proyectar datos en nuevos ejes (componentes principales)")

        # Seleccionar features numéricas
        features = ['pH', 'E.C', 'TDS', 'T.H', 'SAR', 'Ca', 'Mg', 'Na', 'K', 'Cl']
        features_disponibles = [f for f in features if f in self.df_procesado.columns]

        X = self.df_procesado[features_disponibles].dropna()

        # Estandarizar
        X_scaled = self.scaler.fit_transform(X)

        # Aplicar PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Resultados
        print("\n📊 RESULTADOS:")
        print("-" * 80)
        print(f"Variables originales: {len(features_disponibles)}")
        print(f"Componentes principales: {n_components}")

        # Varianza explicada
        print("\n📈 Varianza explicada por componente:")
        for i, var in enumerate(pca.explained_variance_ratio_, 1):
            cum_var = pca.explained_variance_ratio_[:i].sum()
            print(f"   PC{i}: {var*100:.2f}% (acumulado: {cum_var*100:.2f}%)")

        # Loadings (contribuciones)
        print("\n🔍 LOADINGS (Contribución de variables a componentes):")
        print("-" * 80)
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=features_disponibles
        )
        print(loadings.round(3))

        print("\n💡 INTERPRETACIÓN:")
        print("-" * 80)

        # Interpretar PC1
        pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False)
        print(f"\n🎯 PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza):")
        print("   Variables más importantes:")
        for var, loading in pc1_loadings.head(3).items():
            print(f"   • {var}: {loadings.loc[var, 'PC1']:.3f}")
        print("   → PC1 representa principalmente la salinidad/mineralización general")

        # Interpretar PC2
        if n_components >= 2:
            pc2_loadings = loadings['PC2'].abs().sort_values(ascending=False)
            print(f"\n🎯 PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza):")
            print("   Variables más importantes:")
            for var, loading in pc2_loadings.head(3).items():
                print(f"   • {var}: {loadings.loc[var, 'PC2']:.3f}")
            print("   → PC2 representa un contraste químico secundario")

        print("\n✅ CONCLUSIÓN:")
        cum_var = pca.explained_variance_ratio_[:3].sum() * 100
        print(f"   Los primeros 3 componentes capturan {cum_var:.1f}% de la variabilidad.")
        print("   Esto permite reducir de {0} a 3 dimensiones con mínima pérdida de información.".format(len(features_disponibles)))

        # Visualización
        self._plot_pca_biplot(X_pca, loadings, pca, features_disponibles)

        return pca, X_pca, loadings

    def _plot_pca_biplot(self, X_pca, loadings, pca, features):
        """Crea biplot de PCA"""
        fig = go.Figure()

        # Scatter de muestras
        fig.add_trace(go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode='markers',
            marker=dict(size=5, color='lightblue', opacity=0.6),
            name='Muestras',
            text=[f'Muestra {i}' for i in range(len(X_pca))]
        ))

        # Vectores de variables
        scale = 3
        for i, feature in enumerate(features):
            fig.add_trace(go.Scatter(
                x=[0, loadings.iloc[i, 0] * scale],
                y=[0, loadings.iloc[i, 1] * scale],
                mode='lines+text',
                line=dict(color='red', width=2),
                text=['', feature],
                textposition='top center',
                name=feature,
                showlegend=False
            ))

        fig.update_layout(
            title=f"PCA Biplot - PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%) vs PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            width=1000,
            height=800
        )

        fig.write_html("output_pca_biplot.html")
        print("\n📊 Visualización guardada: output_pca_biplot.html")

    def analisis_clustering(self, n_clusters=4):
        """
        Análisis de Clustering (Agrupamiento)

        Algoritmo: K-Means Clustering
        - Agrupa muestras similares en clusters
        - Minimiza distancia intra-cluster
        - Maximiza distancia inter-cluster

        Proceso:
        1. Inicializar k centroides aleatoriamente
        2. Asignar cada punto al centroide más cercano
        3. Recalcular centroides como media del cluster
        4. Repetir 2-3 hasta convergencia

        Interpretación:
        - Cada cluster representa un tipo de agua similar
        - Centroides muestran características promedio del grupo
        - Silhouette score mide calidad del clustering (0-1, mayor es mejor)

        Aplicación:
        - Identificar zonas con calidad de agua similar
        - Clasificar fuentes de agua en categorías
        - Detectar patrones geográficos
        """
        print("\n" + "="*80)
        print("🎯 ANÁLISIS DE CLUSTERING (K-MEANS)")
        print("="*80)

        print("\n📚 ALGORITMO: K-Means Clustering")
        print("-" * 80)
        print("K-Means agrupa muestras en k clusters basándose en similitud.")
        print("\nObjetivo: Minimizar la suma de distancias cuadradas dentro de cada cluster")
        print("Fórmula: Σ Σ ||x - μk||² donde μk es el centroide del cluster k")

        # Preparar datos
        features = ['TDS', 'T.H', 'SAR', 'pH', 'E.C']
        features_disponibles = [f for f in features if f in self.df_procesado.columns]

        X = self.df_procesado[features_disponibles].dropna()
        X_scaled = StandardScaler().fit_transform(X)

        # Aplicar K-Means
        print(f"\n🔄 Ejecutando K-Means con k={n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Calcular métricas
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        silhouette = silhouette_score(X_scaled, clusters)
        davies_bouldin = davies_bouldin_score(X_scaled, clusters)
        inertia = kmeans.inertia_

        print("\n📊 RESULTADOS:")
        print("-" * 80)
        print(f"Número de clusters: {n_clusters}")
        print(f"Silhouette Score: {silhouette:.3f} (0-1, mayor es mejor)")
        print(f"Davies-Bouldin Index: {davies_bouldin:.3f} (menor es mejor)")
        print(f"Inertia: {inertia:.2f} (suma de distancias cuadradas)")

        # Análisis de clusters
        self.df_procesado.loc[X.index, 'cluster'] = clusters

        print("\n🔍 CARACTERÍSTICAS DE CADA CLUSTER:")
        print("-" * 80)

        for i in range(n_clusters):
            cluster_data = self.df_procesado[self.df_procesado['cluster'] == i]
            n_samples = len(cluster_data)
            pct = (n_samples / len(X) * 100)

            print(f"\n📌 CLUSTER {i} ({n_samples} muestras, {pct:.1f}%):")

            # Estadísticas del cluster
            for feature in features_disponibles:
                media = cluster_data[feature].mean()
                std = cluster_data[feature].std()
                print(f"   • {feature}: {media:.2f} ± {std:.2f}")

            # Interpretación
            tds_mean = cluster_data['TDS'].mean() if 'TDS' in cluster_data.columns else 0
            sar_mean = cluster_data['SAR'].mean() if 'SAR' in cluster_data.columns else 0

            if tds_mean < 1000 and sar_mean < 10:
                calidad = "EXCELENTE"
                color = "🟢"
            elif tds_mean < 2000 and sar_mean < 15:
                calidad = "BUENA"
                color = "🟡"
            elif tds_mean < 3000:
                calidad = "MODERADA"
                color = "🟠"
            else:
                calidad = "POBRE"
                color = "🔴"

            print(f"   {color} Calidad: {calidad}")

        print("\n💡 INTERPRETACIÓN:")
        print("-" * 80)
        print(f"   Silhouette Score = {silhouette:.3f}:")
        if silhouette > 0.5:
            print("   ✅ Clusters bien definidos y separados")
        elif silhouette > 0.3:
            print("   ⚠️  Clusters moderadamente definidos")
        else:
            print("   ❌ Clusters poco definidos, considerar diferente k")

        print("\n✅ CONCLUSIÓN:")
        print("   Los clusters identifican grupos naturales de calidad de agua.")
        print("   Útil para zonificación y gestión diferenciada por área.")

        # Visualización
        self._plot_clusters(X_scaled, clusters, kmeans, features_disponibles)

        return kmeans, clusters

    def _plot_clusters(self, X_scaled, clusters, kmeans, features):
        """Visualiza clusters en espacio 2D (PCA)"""
        # Reducir a 2D con PCA para visualización
        pca_viz = PCA(n_components=2)
        X_pca = pca_viz.fit_transform(X_scaled)
        centroids_pca = pca_viz.transform(kmeans.cluster_centers_)

        fig = go.Figure()

        # Puntos coloreados por cluster
        for i in range(kmeans.n_clusters):
            mask = clusters == i
            fig.add_trace(go.Scatter(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(size=6, opacity=0.6)
            ))

        # Centroides
        fig.add_trace(go.Scatter(
            x=centroids_pca[:, 0],
            y=centroids_pca[:, 1],
            mode='markers',
            name='Centroides',
            marker=dict(size=15, color='black', symbol='x', line=dict(width=2))
        ))

        fig.update_layout(
            title=f"K-Means Clustering (k={kmeans.n_clusters}) - Proyección PCA",
            xaxis_title=f"PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}%)",
            width=1000,
            height=700
        )

        fig.write_html("output_clustering.html")
        print("\n📊 Visualización guardada: output_clustering.html")

    def analisis_machine_learning(self):
        """
        Análisis con Machine Learning - Clasificación

        Algoritmos comparados:
        1. Random Forest: Ensemble de árboles de decisión
        2. Gradient Boosting: Boosting secuencial de árboles
        3. SVM: Support Vector Machine con kernel RBF
        4. KNN: K-Nearest Neighbors

        Objetivo: Predecir clasificación de calidad de agua (C1S1, C2S1, etc.)

        Métricas:
        - Accuracy: % de predicciones correctas
        - Precision: De las predichas como clase X, cuántas son realmente X
        - Recall: De las reales clase X, cuántas fueron detectadas
        - F1-Score: Media armónica de Precision y Recall
        """
        print("\n" + "="*80)
        print("🤖 ANÁLISIS CON MACHINE LEARNING")
        print("="*80)

        print("\n📚 ALGORITMOS A COMPARAR:")
        print("-" * 80)
        print("1️⃣  Random Forest:")
        print("   • Ensemble de múltiples árboles de decisión")
        print("   • Cada árbol vota, mayoría gana")
        print("   • Robusto a overfitting, maneja no-linealidad")
        print("\n2️⃣  Gradient Boosting:")
        print("   • Construye árboles secuencialmente")
        print("   • Cada árbol corrige errores del anterior")
        print("   • Alta precisión, puede overfittear")
        print("\n3️⃣  SVM (Support Vector Machine):")
        print("   • Encuentra hiperplano óptimo de separación")
        print("   • Kernel RBF para relaciones no-lineales")
        print("   • Efectivo en espacios de alta dimensión")
        print("\n4️⃣  KNN (K-Nearest Neighbors):")
        print("   • Clasifica según k vecinos más cercanos")
        print("   • Simple, no paramétrico")
        print("   • Sensible a escala de features")

        # Preparar datos
        print("\n🔄 Preparando datos...")

        # Features
        features = ['pH', 'E.C', 'TDS', 'T.H', 'SAR', 'RSC', 'Ca', 'Mg', 'Na', 'K', 'Cl']
        features_disponibles = [f for f in features if f in self.df_procesado.columns]

        # Target
        if 'Classification' not in self.df_procesado.columns:
            print("❌ Columna 'Classification' no encontrada")
            return None

        # Filtrar datos completos
        df_ml = self.df_procesado[features_disponibles + ['Classification']].dropna()

        X = df_ml[features_disponibles]
        y = df_ml['Classification']

        # Agrupar clases minoritarias
        print("\n🔍 Analizando distribución de clases...")
        class_counts = y.value_counts()
        print(f"   Total de clases: {len(class_counts)}")

        # Identificar clases con pocas muestras (< 5)
        clases_minoritarias = class_counts[class_counts < 5].index.tolist()
        if clases_minoritarias:
            print(f"   ⚠️  Clases minoritarias detectadas: {len(clases_minoritarias)}")
            print(f"      {clases_minoritarias}")
            print("   → Agrupando clases minoritarias en 'OTROS'")

            # Agrupar clases minoritarias
            y = y.replace(clases_minoritarias, 'OTROS')
            class_counts = y.value_counts()

        print(f"\n   Distribución final de clases:")
        for clase, count in class_counts.head(10).items():
            pct = (count / len(y) * 100)
            print(f"   • {clase}: {count} ({pct:.1f}%)")

        if len(class_counts) > 10:
            print(f"   ... y {len(class_counts) - 10} clases más")

        # Codificar target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        print(f"\n   ✓ Muestras: {len(X)}")
        print(f"   ✓ Features: {len(features_disponibles)}")
        print(f"   ✓ Clases finales: {len(le.classes_)}")

        # Split train/test (sin estratificación si hay clases muy desbalanceadas)
        # Verificar si es posible estratificar
        min_class_count = class_counts.min()
        use_stratify = min_class_count >= 2

        if use_stratify:
            print("   ✓ Usando división estratificada")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
            )
        else:
            print("   ⚠️  Usando división aleatoria (clases muy desbalanceadas)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42
            )

        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"   ✓ Train: {len(X_train)} muestras")
        print(f"   ✓ Test: {len(X_test)} muestras")

        # Entrenar modelos
        print("\n🎯 ENTRENANDO MODELOS...")
        print("-" * 80)

        modelos = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }

        resultados = {}

        for nombre, modelo in modelos.items():
            print(f"\n🔄 Entrenando {nombre}...")

            # Entrenar
            modelo.fit(X_train_scaled, y_train)

            # Predecir
            y_pred = modelo.predict(X_test_scaled)

            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Cross-validation
            cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5)

            resultados[nombre] = {
                'modelo': modelo,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred
            }

            print(f"   ✓ Accuracy: {accuracy:.3f}")
            print(f"   ✓ Precision: {precision:.3f}")
            print(f"   ✓ Recall: {recall:.3f}")
            print(f"   ✓ F1-Score: {f1:.3f}")
            print(f"   ✓ CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Comparación
        print("\n📊 COMPARACIÓN DE MODELOS:")
        print("-" * 80)

        df_resultados = pd.DataFrame({
            'Modelo': list(resultados.keys()),
            'Accuracy': [r['accuracy'] for r in resultados.values()],
            'Precision': [r['precision'] for r in resultados.values()],
            'Recall': [r['recall'] for r in resultados.values()],
            'F1-Score': [r['f1'] for r in resultados.values()],
            'CV Score': [r['cv_mean'] for r in resultados.values()]
        })

        df_resultados = df_resultados.sort_values('F1-Score', ascending=False)
        print(df_resultados.to_string(index=False))

        # Mejor modelo
        mejor_modelo_nombre = df_resultados.iloc[0]['Modelo']
        mejor_score = df_resultados.iloc[0]['F1-Score']

        print("\n💡 INTERPRETACIÓN:")
        print("-" * 80)
        print(f"🏆 Mejor modelo: {mejor_modelo_nombre} (F1-Score: {mejor_score:.3f})")

        if mejor_score > 0.8:
            print("   ✅ Excelente capacidad predictiva (>80%)")
        elif mejor_score > 0.6:
            print("   ⚠️  Capacidad predictiva moderada (60-80%)")
        else:
            print("   ❌ Capacidad predictiva limitada (<60%)")

        print("\n📈 Significado de las métricas:")
        print("   • Accuracy: % total de aciertos")
        print("   • Precision: Confiabilidad de predicciones positivas")
        print("   • Recall: Capacidad de detectar todas las instancias")
        print("   • F1-Score: Balance entre Precision y Recall")

        print("\n✅ CONCLUSIÓN:")
        print(f"   {mejor_modelo_nombre} es el más adecuado para clasificar calidad de agua.")
        print("   Puede usarse para predecir clasificación de nuevas muestras.")

        # Matriz de confusión del mejor modelo
        self._plot_confusion_matrix(
            y_test,
            resultados[mejor_modelo_nombre]['y_pred'],
            le.classes_,
            mejor_modelo_nombre
        )

        # Feature importance (si es Random Forest o Gradient Boosting)
        if mejor_modelo_nombre in ['Random Forest', 'Gradient Boosting']:
            self._plot_feature_importance(
                resultados[mejor_modelo_nombre]['modelo'],
                features_disponibles,
                mejor_modelo_nombre
            )

        self.modelos = resultados
        return resultados, le

    def _plot_confusion_matrix(self, y_true, y_pred, classes, model_name):
        """Visualiza matriz de confusión"""
        cm = confusion_matrix(y_true, y_pred)

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Frecuencia")
        ))

        fig.update_layout(
            title=f"Matriz de Confusión - {model_name}",
            xaxis_title="Predicción",
            yaxis_title="Real",
            width=800,
            height=700
        )

        fig.write_html(f"output_confusion_matrix_{model_name.replace(' ', '_')}.html")
        print(f"\n📊 Matriz de confusión guardada: output_confusion_matrix_{model_name.replace(' ', '_')}.html")

    def _plot_feature_importance(self, modelo, features, model_name):
        """Visualiza importancia de features"""
        importances = modelo.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig = go.Figure(data=[
            go.Bar(
                x=[features[i] for i in indices],
                y=[importances[i] for i in indices],
                marker_color='steelblue'
            )
        ])

        fig.update_layout(
            title=f"Importancia de Features - {model_name}",
            xaxis_title="Parámetro",
            yaxis_title="Importancia",
            width=1000,
            height=600
        )

        fig.write_html(f"output_feature_importance_{model_name.replace(' ', '_')}.html")
        print(f"📊 Feature importance guardada: output_feature_importance_{model_name.replace(' ', '_')}.html")

    def sistema_ranking_calidad(self):
        """
        Sistema de Ranking de Calidad de Agua

        Metodología:
        - Scoring multi-criterio basado en normativas
        - Ponderación de parámetros según importancia
        - Clasificación en 5 niveles de calidad

        Criterios evaluados:
        1. TDS (40%): Sólidos disueltos totales
        2. SAR (25%): Relación de absorción de sodio
        3. RSC (20%): Carbonato de sodio residual
        4. pH (10%): Acidez/alcalinidad
        5. Dureza (5%): Contenido de Ca y Mg

        Interpretación:
        - Score 80-100: Excelente (uso sin restricciones)
        - Score 60-80: Buena (uso con precauciones menores)
        - Score 40-60: Moderada (requiere manejo)
        - Score 20-40: Pobre (uso limitado)
        - Score 0-20: Muy pobre (no recomendado)
        """
        print("\n" + "="*80)
        print("🏆 SISTEMA DE RANKING DE CALIDAD DE AGUA")
        print("="*80)

        print("\n📚 METODOLOGÍA DE SCORING:")
        print("-" * 80)
        print("Sistema multi-criterio con ponderación basada en impacto agronómico")
        print("\nPonderaciones:")
        print("   • TDS (40%): Impacto en salinidad del suelo")
        print("   • SAR (25%): Riesgo de sodificación")
        print("   • RSC (20%): Alcalinidad residual")
        print("   • pH (10%): Acidez/alcalinidad")
        print("   • Dureza (5%): Incrustaciones y manejo")

        # Calcular scores individuales
        df_score = self.df_procesado.copy()

        # Score TDS (0-100, menor es mejor)
        if 'TDS' in df_score.columns:
            df_score['score_tds'] = 100 - np.clip((df_score['TDS'] / 3000) * 100, 0, 100)
        else:
            df_score['score_tds'] = 50

        # Score SAR (0-100, menor es mejor)
        if 'SAR' in df_score.columns:
            df_score['score_sar'] = 100 - np.clip((df_score['SAR'] / 26) * 100, 0, 100)
        else:
            df_score['score_sar'] = 50

        # Score RSC (0-100, menor es mejor)
        if 'RSC' in df_score.columns:
            df_score['score_rsc'] = 100 - np.clip((df_score['RSC'] / 2.5) * 100, 0, 100)
        else:
            df_score['score_rsc'] = 50

        # Score pH (0-100, óptimo 6.5-7.5)
        if 'pH' in df_score.columns:
            df_score['score_ph'] = 100 - np.abs(df_score['pH'] - 7.0) * 10
            df_score['score_ph'] = np.clip(df_score['score_ph'], 0, 100)
        else:
            df_score['score_ph'] = 50

        # Score Dureza (0-100, menor es mejor)
        if 'T.H' in df_score.columns:
            df_score['score_dureza'] = 100 - np.clip((df_score['T.H'] / 500) * 100, 0, 100)
        else:
            df_score['score_dureza'] = 50

        # Score total ponderado
        df_score['quality_score'] = (
            df_score['score_tds'] * 0.40 +
            df_score['score_sar'] * 0.25 +
            df_score['score_rsc'] * 0.20 +
            df_score['score_ph'] * 0.10 +
            df_score['score_dureza'] * 0.05
        )

        # Clasificación
        def clasificar_calidad(score):
            if score >= 80:
                return 'Excelente'
            elif score >= 60:
                return 'Buena'
            elif score >= 40:
                return 'Moderada'
            elif score >= 20:
                return 'Pobre'
            else:
                return 'Muy Pobre'

        df_score['quality_category'] = df_score['quality_score'].apply(clasificar_calidad)

        # Resultados
        print("\n📊 DISTRIBUCIÓN DE CALIDAD:")
        print("-" * 80)

        dist = df_score['quality_category'].value_counts()
        for categoria in ['Excelente', 'Buena', 'Moderada', 'Pobre', 'Muy Pobre']:
            if categoria in dist.index:
                count = dist[categoria]
                pct = (count / len(df_score) * 100)

                if categoria == 'Excelente':
                    emoji = "🟢"
                elif categoria == 'Buena':
                    emoji = "🔵"
                elif categoria == 'Moderada':
                    emoji = "🟡"
                elif categoria == 'Pobre':
                    emoji = "🟠"
                else:
                    emoji = "🔴"

                print(f"   {emoji} {categoria}: {count} muestras ({pct:.1f}%)")

        # Top 10 mejores fuentes
        print("\n🏆 TOP 10 MEJORES FUENTES DE AGUA:")
        print("-" * 80)

        top10 = df_score.nlargest(10, 'quality_score')

        for idx, (_, row) in enumerate(top10.iterrows(), 1):
            score = row['quality_score']
            distrito = row.get('district', 'N/A')
            mandal = row.get('mandal', 'N/A')
            village = row.get('village', 'N/A')

            print(f"\n{idx}. Score: {score:.1f}/100 - {row['quality_category']}")
            print(f"   📍 Ubicación: {village}, {mandal}, {distrito}")

            if 'TDS' in row:
                print(f"   💧 TDS: {row['TDS']:.0f} mg/L")
            if 'SAR' in row:
                print(f"   🧪 SAR: {row['SAR']:.2f}")
            if 'pH' in row:
                print(f"   ⚗️  pH: {row['pH']:.2f}")

        print("\n💡 INTERPRETACIÓN:")
        print("-" * 80)
        print("   Score 80-100: Uso sin restricciones para riego y consumo animal")
        print("   Score 60-80: Uso con precauciones menores (monitoreo de salinidad)")
        print("   Score 40-60: Requiere manejo (cultivos tolerantes, drenaje)")
        print("   Score 20-40: Uso limitado (solo cultivos muy tolerantes)")
        print("   Score 0-20: No recomendado (riesgo de degradación del suelo)")

        print("\n✅ CONCLUSIÓN:")
        excelentes = len(df_score[df_score['quality_category'] == 'Excelente'])
        buenas = len(df_score[df_score['quality_category'] == 'Buena'])
        total_aceptable = excelentes + buenas
        pct_aceptable = (total_aceptable / len(df_score) * 100)

        print(f"   {pct_aceptable:.1f}% de las fuentes tienen calidad aceptable (Excelente/Buena)")
        print(f"   Las mejores fuentes están en: {top10['district'].mode()[0] if 'district' in top10.columns else 'N/A'}")

        # Guardar resultados
        self.df_procesado = df_score

        # Visualización
        self._plot_quality_distribution(df_score)
        self._plot_quality_map(df_score)

        return df_score[['quality_score', 'quality_category']].describe()

    def _plot_quality_distribution(self, df_score):
        """Visualiza distribución de calidad"""
        fig = go.Figure()

        # Histograma
        fig.add_trace(go.Histogram(
            x=df_score['quality_score'],
            nbinsx=20,
            name='Distribución',
            marker_color='steelblue'
        ))

        # Líneas de referencia
        fig.add_vline(x=80, line_dash="dash", line_color="green", annotation_text="Excelente")
        fig.add_vline(x=60, line_dash="dash", line_color="blue", annotation_text="Buena")
        fig.add_vline(x=40, line_dash="dash", line_color="orange", annotation_text="Moderada")
        fig.add_vline(x=20, line_dash="dash", line_color="red", annotation_text="Pobre")

        fig.update_layout(
            title="Distribución de Quality Score",
            xaxis_title="Quality Score (0-100)",
            yaxis_title="Frecuencia",
            width=1000,
            height=600
        )

        fig.write_html("output_quality_distribution.html")
        print("\n📊 Distribución guardada: output_quality_distribution.html")

    def _plot_quality_map(self, df_score):
        """Mapa de calidad geográfica"""
        if 'lat_gis' not in df_score.columns or 'long_gis' not in df_score.columns:
            return

        # Filtrar datos válidos
        df_map = df_score[
            (df_score['lat_gis'].notna()) &
            (df_score['long_gis'].notna()) &
            (df_score['lat_gis'] != 0) &
            (df_score['long_gis'] != 0)
        ].copy()

        if len(df_map) == 0:
            return

        # Mapa de calor
        fig = px.scatter_mapbox(
            df_map,
            lat='lat_gis',
            lon='long_gis',
            color='quality_score',
            size='quality_score',
            color_continuous_scale='RdYlGn',
            size_max=15,
            zoom=6,
            mapbox_style="open-street-map",
            hover_data=['district', 'quality_category', 'TDS', 'SAR'] if 'TDS' in df_map.columns else ['district', 'quality_category'],
            title="Mapa de Calidad de Agua - Telangana"
        )

        fig.update_layout(width=1200, height=800)
        fig.write_html("output_quality_map.html")
        print("📊 Mapa de calidad guardado: output_quality_map.html")

    def dashboard_ejecutivo(self):
        """
        Dashboard Ejecutivo Interactivo

        Genera visualizaciones comprehensivas para toma de decisiones:
        1. KPIs principales
        2. Tendencias temporales
        3. Distribución espacial
        4. Comparación entre distritos
        5. Análisis de parámetros críticos
        """
        print("\n" + "="*80)
        print("📊 GENERANDO DASHBOARD EJECUTIVO")
        print("="*80)

        # Crear dashboard con subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Distribución de Calidad',
                'TDS por Distrito (Top 10)',
                'Evolución Temporal de pH',
                'SAR vs TDS',
                'Clasificación de Agua',
                'Parámetros Críticos'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'box'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )

        # 1. Distribución de calidad
        if 'quality_category' in self.df_procesado.columns:
            quality_dist = self.df_procesado['quality_category'].value_counts()
            fig.add_trace(
                go.Bar(x=quality_dist.index, y=quality_dist.values, marker_color='steelblue'),
                row=1, col=1
            )

        # 2. TDS por distrito (top 10)
        if 'TDS' in self.df_procesado.columns and 'district' in self.df_procesado.columns:
            top_districts = self.df_procesado.groupby('district')['TDS'].mean().nlargest(10)
            for distrito in top_districts.index:
                data = self.df_procesado[self.df_procesado['district'] == distrito]['TDS']
                fig.add_trace(
                    go.Box(y=data, name=distrito, showlegend=False),
                    row=1, col=2
                )

        # 3. Evolución temporal de pH
        if 'pH' in self.df_procesado.columns and 'año' in self.df_procesado.columns:
            ph_temporal = self.df_procesado.groupby('año')['pH'].agg(['mean', 'std']).reset_index()
            fig.add_trace(
                go.Scatter(
                    x=ph_temporal['año'],
                    y=ph_temporal['mean'],
                    mode='lines+markers',
                    name='pH promedio',
                    error_y=dict(type='data', array=ph_temporal['std']),
                    showlegend=False
                ),
                row=2, col=1
            )

        # 4. SAR vs TDS
        if 'SAR' in self.df_procesado.columns and 'TDS' in self.df_procesado.columns:
            sample = self.df_procesado.sample(min(500, len(self.df_procesado)))
            fig.add_trace(
                go.Scatter(
                    x=sample['TDS'],
                    y=sample['SAR'],
                    mode='markers',
                    marker=dict(size=5, opacity=0.6),
                    showlegend=False
                ),
                row=2, col=2
            )

        # 5. Clasificación de agua
        if 'Classification' in self.df_procesado.columns:
            class_dist = self.df_procesado['Classification'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=class_dist.index, y=class_dist.values, marker_color='coral'),
                row=3, col=1
            )

        # 6. Parámetros críticos (promedio)
        params = ['pH', 'TDS', 'SAR', 'RSC', 'T.H']
        params_disponibles = [p for p in params if p in self.df_procesado.columns]
        if params_disponibles:
            # Normalizar para comparación
            valores_norm = []
            for param in params_disponibles:
                val = self.df_procesado[param].mean()
                max_val = self.df_procesado[param].max()
                valores_norm.append((val / max_val) * 100 if max_val > 0 else 0)

            fig.add_trace(
                go.Bar(x=params_disponibles, y=valores_norm, marker_color='lightgreen'),
                row=3, col=2
            )

        # Layout
        fig.update_layout(
            title_text="Dashboard Ejecutivo - Calidad de Agua Subterránea Telangana",
            showlegend=False,
            height=1200,
            width=1400
        )

        # Actualizar ejes
        fig.update_xaxes(title_text="Categoría", row=1, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=1, col=1)

        fig.update_xaxes(title_text="Distrito", row=1, col=2)
        fig.update_yaxes(title_text="TDS (mg/L)", row=1, col=2)

        fig.update_xaxes(title_text="Año", row=2, col=1)
        fig.update_yaxes(title_text="pH", row=2, col=1)

        fig.update_xaxes(title_text="TDS (mg/L)", row=2, col=2)
        fig.update_yaxes(title_text="SAR", row=2, col=2)

        fig.update_xaxes(title_text="Clasificación", row=3, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=3, col=1)

        fig.update_xaxes(title_text="Parámetro", row=3, col=2)
        fig.update_yaxes(title_text="Valor Normalizado (%)", row=3, col=2)

        fig.write_html("output_dashboard_ejecutivo.html")

        print("\n✅ Dashboard ejecutivo generado: output_dashboard_ejecutivo.html")
        print("\n📊 El dashboard incluye:")
        print("   • Distribución de calidad de agua")
        print("   • Análisis por distrito")
        print("   • Tendencias temporales")
        print("   • Relaciones entre parámetros")
        print("   • Clasificaciones y parámetros críticos")

        return fig

    def reporte_ejecutivo_final(self):
        """
        Reporte Ejecutivo Final

        Resumen comprehensivo con conclusiones y recomendaciones
        """
        print("\n" + "="*80)
        print("📋 REPORTE EJECUTIVO FINAL")
        print("="*80)

        print("\n🎯 OBJETIVO DEL ANÁLISIS:")
        print("-" * 80)
        print("Identificar las mejores fuentes de agua subterránea en Telangana")
        print("para uso agrícola y consumo animal, basado en análisis multi-criterio.")

        print("\n📊 RESUMEN DE DATOS:")
        print("-" * 80)
        print(f"   • Total de muestras analizadas: {len(self.df)}")
        print(f"   • Período de análisis: {self.df['año'].min()}-{self.df['año'].max()}")
        print(f"   • Distritos cubiertos: {self.df['district'].nunique()}")
        print(f"   • Parámetros evaluados: {len(self.df.select_dtypes(include=[np.number]).columns)}")

        print("\n🔬 METODOLOGÍA APLICADA:")
        print("-" * 80)
        print("   ✓ Análisis Estadístico Descriptivo (Pandas, NumPy)")
        print("   ✓ Análisis de Correlación (Pearson)")
        print("   ✓ PCA - Reducción de Dimensionalidad")
        print("   ✓ K-Means Clustering - Agrupamiento")
        print("   ✓ Machine Learning - Clasificación (Random Forest, SVM, etc.)")
        print("   ✓ Sistema de Scoring Multi-criterio")

        print("\n🏆 HALLAZGOS PRINCIPALES:")
        print("-" * 80)

        if 'quality_category' in self.df_procesado.columns:
            # Distribución de calidad
            dist = self.df_procesado['quality_category'].value_counts()
            total = len(self.df_procesado)

            print("\n1️⃣  DISTRIBUCIÓN DE CALIDAD:")
            for categoria in ['Excelente', 'Buena', 'Moderada', 'Pobre', 'Muy Pobre']:
                if categoria in dist.index:
                    count = dist[categoria]
                    pct = (count / total * 100)
                    print(f"   • {categoria}: {count} fuentes ({pct:.1f}%)")

            # Mejores distritos
            if 'district' in self.df_procesado.columns and 'quality_score' in self.df_procesado.columns:
                print("\n2️⃣  MEJORES DISTRITOS (por quality score promedio):")
                top_districts = self.df_procesado.groupby('district')['quality_score'].mean().nlargest(5)
                for i, (distrito, score) in enumerate(top_districts.items(), 1):
                    print(f"   {i}. {distrito}: {score:.1f}/100")

            # Parámetros críticos
            print("\n3️⃣  PARÁMETROS CRÍTICOS (promedios):")
            if 'TDS' in self.df_procesado.columns:
                tds_mean = self.df_procesado['TDS'].mean()
                print(f"   • TDS: {tds_mean:.0f} mg/L", end="")
                if tds_mean < 1000:
                    print(" ✅ (Excelente)")
                elif tds_mean < 2000:
                    print(" ⚠️  (Aceptable)")
                else:
                    print(" ❌ (Alto)")

            if 'SAR' in self.df_procesado.columns:
                sar_mean = self.df_procesado['SAR'].mean()
                print(f"   • SAR: {sar_mean:.2f}", end="")
                if sar_mean < 10:
                    print(" ✅ (Bajo riesgo)")
                elif sar_mean < 18:
                    print(" ⚠️  (Riesgo medio)")
                else:
                    print(" ❌ (Alto riesgo)")

            if 'pH' in self.df_procesado.columns:
                ph_mean = self.df_procesado['pH'].mean()
                print(f"   • pH: {ph_mean:.2f}", end="")
                if 6.5 <= ph_mean <= 8.5:
                    print(" ✅ (Rango óptimo)")
                else:
                    print(" ⚠️  (Fuera de rango óptimo)")

        print("\n💡 RECOMENDACIONES:")
        print("-" * 80)

        if 'quality_score' in self.df_procesado.columns:
            excelentes = len(self.df_procesado[self.df_procesado['quality_category'] == 'Excelente'])
            buenas = len(self.df_procesado[self.df_procesado['quality_category'] == 'Buena'])

            if excelentes > 0:
                print(f"\n✅ FUENTES EXCELENTES ({excelentes} identificadas):")
                print("   • Uso sin restricciones para riego")
                print("   • Apto para consumo animal")
                print("   • Bajo riesgo de salinización")
                print("   • Priorizar para desarrollo agrícola")

            if buenas > 0:
                print(f"\n🔵 FUENTES BUENAS ({buenas} identificadas):")
                print("   • Uso con precauciones menores")
                print("   • Monitoreo periódico de salinidad")
                print("   • Rotación de cultivos recomendada")

            moderadas = len(self.df_procesado[self.df_procesado['quality_category'] == 'Moderada'])
            if moderadas > 0:
                print(f"\n⚠️  FUENTES MODERADAS ({moderadas} identificadas):")
                print("   • Requiere manejo cuidadoso")
                print("   • Cultivos tolerantes a sal")
                print("   • Sistema de drenaje necesario")
                print("   • Enmiendas de suelo (yeso)")

            pobres = len(self.df_procesado[self.df_procesado['quality_category'].isin(['Pobre', 'Muy Pobre'])])
            if pobres > 0:
                print(f"\n❌ FUENTES POBRES/MUY POBRES ({pobres} identificadas):")
                print("   • Uso limitado o no recomendado")
                print("   • Alto riesgo de degradación del suelo")
                print("   • Considerar tratamiento o fuentes alternativas")

        print("\n🎯 CONCLUSIONES FINALES:")
        print("-" * 80)

        if 'quality_score' in self.df_procesado.columns:
            score_promedio = self.df_procesado['quality_score'].mean()

            print(f"\n1. Quality Score Promedio: {score_promedio:.1f}/100")

            if score_promedio >= 60:
                print("   ✅ La calidad general del agua subterránea es ACEPTABLE")
            else:
                print("   ⚠️  La calidad general del agua subterránea requiere ATENCIÓN")

            print("\n2. Variabilidad Espacial:")
            if 'district' in self.df_procesado.columns:
                variabilidad = self.df_procesado.groupby('district')['quality_score'].std().mean()
                print(f"   • Desviación estándar promedio entre distritos: {variabilidad:.1f}")
                if variabilidad > 15:
                    print("   → Alta variabilidad: gestión diferenciada por zona")
                else:
                    print("   → Variabilidad moderada: políticas generales aplicables")

            print("\n3. Tendencias Temporales:")
            if 'año' in self.df_procesado.columns:
                trend = self.df_procesado.groupby('año')['quality_score'].mean()
                if len(trend) > 1:
                    cambio = trend.iloc[-1] - trend.iloc[0]
                    if cambio > 5:
                        print(f"   ✅ Mejora de {cambio:.1f} puntos ({trend.index[0]}-{trend.index[-1]})")
                    elif cambio < -5:
                        print(f"   ⚠️  Deterioro de {abs(cambio):.1f} puntos ({trend.index[0]}-{trend.index[-1]})")
                    else:
                        print(f"   → Estable (cambio de {cambio:.1f} puntos)")

        print("\n📈 PRÓXIMOS PASOS SUGERIDOS:")
        print("-" * 80)
        print("   1. Monitoreo continuo de fuentes clasificadas como 'Moderada' o inferior")
        print("   2. Implementar prácticas de manejo según clasificación")
        print("   3. Desarrollar programa de mejoramiento para zonas críticas")
        print("   4. Establecer red de monitoreo en tiempo real")
        print("   5. Capacitación a agricultores sobre uso adecuado según calidad")

        print("\n" + "="*80)
        print("✅ ANÁLISIS COMPLETADO")
        print("="*80)
        print("\n📊 Archivos generados:")
        print("   • output_correlacion_heatmap.html")
        print("   • output_pca_biplot.html")
        print("   • output_clustering.html")
        print("   • output_confusion_matrix_*.html")
        print("   • output_feature_importance_*.html")
        print("   • output_quality_distribution.html")
        print("   • output_quality_map.html")
        print("   • output_dashboard_ejecutivo.html")
        print("\n💡 Abre los archivos HTML en tu navegador para visualizaciones interactivas")


# ============================================================================
# FUNCIÓN PRINCIPAL Y MENÚ
# ============================================================================

def main():
    """Función principal con menú interactivo"""

    analizador = AnalizadorCalidadAgua("samples")

    # Menú principal
    while True:
        print("\n" + "="*80)
        print("MENÚ PRINCIPAL - ANÁLISIS PROFESIONAL DE CALIDAD DE AGUA")
        print("="*80)

        print("\n📊 ANÁLISIS ESTADÍSTICO")
        print("   1. Cargar datos y análisis exploratorio (EDA)")
        print("   2. Análisis de valores faltantes")
        print("   3. Preprocesamiento de datos")
        print("   4. Análisis de correlación")

        print("\n🔬 ANÁLISIS AVANZADO")
        print("   5. PCA - Análisis de Componentes Principales")
        print("   6. Clustering - K-Means")

        print("\n🤖 MACHINE LEARNING")
        print("   7. Comparación de modelos de clasificación")

        print("\n🏆 SISTEMA DE CALIDAD")
        print("   8. Sistema de ranking de calidad de agua")
        print("   9. Dashboard ejecutivo interactivo")

        print("\n📋 REPORTES")
        print("   10. Reporte ejecutivo final")
        print("   11. Análisis completo (ejecutar todo)")

        print("\n   0. Salir")

        try:
            opcion = input("\n👉 Selecciona una opción: ").strip()

            if opcion == "0":
                print("\n✅ ¡Hasta pronto!")
                break

            elif opcion == "1":
                analizador.cargar_datos()
                analizador.analisis_exploratorio()

            elif opcion == "2":
                if analizador.df is None:
                    analizador.cargar_datos()
                analizador.analisis_valores_faltantes()

            elif opcion == "3":
                if analizador.df is None:
                    analizador.cargar_datos()
                analizador.preprocesar_datos()

            elif opcion == "4":
                if analizador.df_procesado is None:
                    analizador.cargar_datos()
                    analizador.preprocesar_datos()
                analizador.analisis_correlacion()

            elif opcion == "5":
                if analizador.df_procesado is None:
                    analizador.cargar_datos()
                    analizador.preprocesar_datos()
                analizador.analisis_pca()

            elif opcion == "6":
                if analizador.df_procesado is None:
                    analizador.cargar_datos()
                    analizador.preprocesar_datos()
                analizador.analisis_clustering()

            elif opcion == "7":
                if analizador.df_procesado is None:
                    analizador.cargar_datos()
                    analizador.preprocesar_datos()
                analizador.analisis_machine_learning()

            elif opcion == "8":
                if analizador.df_procesado is None:
                    analizador.cargar_datos()
                    analizador.preprocesar_datos()
                analizador.sistema_ranking_calidad()

            elif opcion == "9":
                if analizador.df_procesado is None:
                    analizador.cargar_datos()
                    analizador.preprocesar_datos()
                    analizador.sistema_ranking_calidad()
                analizador.dashboard_ejecutivo()

            elif opcion == "10":
                if analizador.df_procesado is None:
                    analizador.cargar_datos()
                    analizador.preprocesar_datos()
                    analizador.sistema_ranking_calidad()
                analizador.reporte_ejecutivo_final()

            elif opcion == "11":
                print("\n🚀 EJECUTANDO ANÁLISIS COMPLETO...")
                print("   Esto puede tomar varios minutos...")

                # Ejecutar todo en secuencia
                analizador.cargar_datos()
                analizador.analisis_exploratorio()
                analizador.analisis_valores_faltantes()
                analizador.preprocesar_datos()
                analizador.analisis_correlacion()
                analizador.analisis_pca()
                analizador.analisis_clustering()
                analizador.analisis_machine_learning()
                analizador.sistema_ranking_calidad()
                analizador.dashboard_ejecutivo()
                analizador.reporte_ejecutivo_final()

                print("\n✅ ANÁLISIS COMPLETO FINALIZADO")

            else:
                print("\n⚠️  Opción no válida")

            input("\n⏸️  Presiona Enter para continuar...")

        except KeyboardInterrupt:
            print("\n\n✅ ¡Hasta pronto!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    main()

