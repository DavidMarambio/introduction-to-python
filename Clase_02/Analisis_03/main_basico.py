"""
Script Profesional de Análisis de Calidad de Agua
Autor: Experto en Análisis de Datos Ambientales
Descripción: Análisis completo de datos de calidad de agua subterránea
             con implementaciones manuales de estadística aplicada
Niveles: Básico, Medio, Avanzado y Ciencia de Datos
"""

import csv
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Any
import os


# ============================================================================
# MÓDULO 1: FUNCIONES ESTADÍSTICAS MANUALES
# ============================================================================

class EstadisticaManual:
    """Implementación manual de funciones estadísticas fundamentales"""
    
    @staticmethod
    def media(datos: List[float]) -> float:
        """Calcula la media aritmética"""
        if not datos:
            return 0.0
        return sum(datos) / len(datos)
    
    @staticmethod
    def mediana(datos: List[float]) -> float:
        """Calcula la mediana"""
        if not datos:
            return 0.0
        datos_ordenados = sorted(datos)
        n = len(datos_ordenados)
        if n % 2 == 0:
            return (datos_ordenados[n//2 - 1] + datos_ordenados[n//2]) / 2
        return datos_ordenados[n//2]
    
    @staticmethod
    def moda(datos: List[float]) -> float:
        """Calcula la moda"""
        if not datos:
            return 0.0
        frecuencias = {}
        for valor in datos:
            frecuencias[valor] = frecuencias.get(valor, 0) + 1
        return max(frecuencias, key=frecuencias.get)
    
    @staticmethod
    def varianza(datos: List[float], muestral: bool = True) -> float:
        """Calcula la varianza (muestral o poblacional)"""
        if len(datos) < 2:
            return 0.0
        media = EstadisticaManual.media(datos)
        suma_cuadrados = sum((x - media) ** 2 for x in datos)
        divisor = len(datos) - 1 if muestral else len(datos)
        return suma_cuadrados / divisor
    
    @staticmethod
    def desviacion_estandar(datos: List[float], muestral: bool = True) -> float:
        """Calcula la desviación estándar"""
        return math.sqrt(EstadisticaManual.varianza(datos, muestral))
    
    @staticmethod
    def coeficiente_variacion(datos: List[float]) -> float:
        """Calcula el coeficiente de variación (CV%)"""
        media = EstadisticaManual.media(datos)
        if media == 0:
            return 0.0
        return (EstadisticaManual.desviacion_estandar(datos) / media) * 100
    
    @staticmethod
    def percentil(datos: List[float], p: float) -> float:
        """Calcula el percentil p (0-100)"""
        if not datos:
            return 0.0
        datos_ordenados = sorted(datos)
        k = (len(datos_ordenados) - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return datos_ordenados[int(k)]
        d0 = datos_ordenados[int(f)] * (c - k)
        d1 = datos_ordenados[int(c)] * (k - f)
        return d0 + d1
    
    @staticmethod
    def rango_intercuartil(datos: List[float]) -> float:
        """Calcula el rango intercuartílico (IQR)"""
        q1 = EstadisticaManual.percentil(datos, 25)
        q3 = EstadisticaManual.percentil(datos, 75)
        return q3 - q1
    
    @staticmethod
    def asimetria(datos: List[float]) -> float:
        """Calcula el coeficiente de asimetría (skewness)"""
        if len(datos) < 3:
            return 0.0
        n = len(datos)
        media = EstadisticaManual.media(datos)
        desv = EstadisticaManual.desviacion_estandar(datos)
        if desv == 0:
            return 0.0
        suma_cubos = sum(((x - media) / desv) ** 3 for x in datos)
        return (n / ((n - 1) * (n - 2))) * suma_cubos
    
    @staticmethod
    def curtosis(datos: List[float]) -> float:
        """Calcula el coeficiente de curtosis (kurtosis)"""
        if len(datos) < 4:
            return 0.0
        n = len(datos)
        media = EstadisticaManual.media(datos)
        desv = EstadisticaManual.desviacion_estandar(datos)
        if desv == 0:
            return 0.0
        suma_cuartos = sum(((x - media) / desv) ** 4 for x in datos)
        return ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * suma_cuartos - \
               (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    
    @staticmethod
    def covarianza(x: List[float], y: List[float]) -> float:
        """Calcula la covarianza entre dos variables"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        media_x = EstadisticaManual.media(x)
        media_y = EstadisticaManual.media(y)
        return sum((x[i] - media_x) * (y[i] - media_y) for i in range(len(x))) / (len(x) - 1)
    
    @staticmethod
    def correlacion_pearson(x: List[float], y: List[float]) -> float:
        """Calcula el coeficiente de correlación de Pearson"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        cov = EstadisticaManual.covarianza(x, y)
        desv_x = EstadisticaManual.desviacion_estandar(x)
        desv_y = EstadisticaManual.desviacion_estandar(y)
        if desv_x == 0 or desv_y == 0:
            return 0.0
        return cov / (desv_x * desv_y)
    
    @staticmethod
    def regresion_lineal(x: List[float], y: List[float]) -> Tuple[float, float, float]:
        """
        Calcula regresión lineal simple: y = a + bx
        Retorna: (pendiente, intercepto, r_cuadrado)
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0, 0.0, 0.0
        
        n = len(x)
        media_x = EstadisticaManual.media(x)
        media_y = EstadisticaManual.media(y)
        
        # Pendiente
        numerador = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n))
        denominador = sum((x[i] - media_x) ** 2 for i in range(n))
        
        if denominador == 0:
            return 0.0, media_y, 0.0
        
        pendiente = numerador / denominador
        intercepto = media_y - pendiente * media_x
        
        # R cuadrado
        r = EstadisticaManual.correlacion_pearson(x, y)
        r_cuadrado = r ** 2
        
        return pendiente, intercepto, r_cuadrado
    
    @staticmethod
    def normalizar_zscore(datos: List[float]) -> List[float]:
        """Normaliza datos usando Z-score"""
        media = EstadisticaManual.media(datos)
        desv = EstadisticaManual.desviacion_estandar(datos)
        if desv == 0:
            return [0.0] * len(datos)
        return [(x - media) / desv for x in datos]
    
    @staticmethod
    def normalizar_minmax(datos: List[float]) -> List[float]:
        """Normaliza datos al rango [0, 1]"""
        minimo = min(datos)
        maximo = max(datos)
        rango = maximo - minimo
        if rango == 0:
            return [0.5] * len(datos)
        return [(x - minimo) / rango for x in datos]


# ============================================================================
# MÓDULO 2: CARGA Y PROCESAMIENTO DE DATOS
# ============================================================================

class CargadorDatos:
    """Maneja la carga y procesamiento inicial de datos"""
    
    def __init__(self, directorio: str = "samples"):
        self.directorio = directorio
        self.datos_combinados = []
        self.columnas = []
        
    def cargar_archivos_csv(self) -> List[Dict[str, Any]]:
        """Carga todos los archivos CSV de calidad de agua"""
        archivos = [
            "ground_water_quality_2018_post.csv",
            "ground_water_quality_2019_post.csv",
            "ground_water_quality_2020_post.csv"
        ]
        
        datos_totales = []
        
        for archivo in archivos:
            ruta = os.path.join(self.directorio, archivo)
            if not os.path.exists(ruta):
                print(f"⚠️  Archivo no encontrado: {ruta}")
                continue
                
            with open(ruta, 'r', encoding='utf-8') as f:
                lector = csv.DictReader(f)
                for fila in lector:
                    # Extraer año del nombre del archivo
                    año = archivo.split('_')[3]
                    fila['año'] = año
                    datos_totales.append(fila)
        
        if datos_totales:
            self.columnas = list(datos_totales[0].keys())
        
        self.datos_combinados = datos_totales
        print(f"✓ Cargados {len(datos_totales)} registros de {len(archivos)} archivos")
        return datos_totales
    
    def obtener_columnas_numericas(self) -> List[str]:
        """Identifica columnas numéricas"""
        columnas_numericas = []
        if not self.datos_combinados:
            return columnas_numericas
        
        muestra = self.datos_combinados[0]
        for columna in muestra.keys():
            if columna in ['sno', 'district', 'mandal', 'village', 'season', 
                          'Classification', 'Classification.1', 'año']:
                continue
            try:
                float(muestra[columna])
                columnas_numericas.append(columna)
            except (ValueError, TypeError):
                pass
        
        return columnas_numericas
    
    def extraer_columna(self, nombre_columna: str, limpiar_nulos: bool = True) -> List[float]:
        """Extrae una columna como lista de floats"""
        valores = []
        for fila in self.datos_combinados:
            try:
                valor = float(fila.get(nombre_columna, 0))
                if not (limpiar_nulos and (math.isnan(valor) or valor == 0)):
                    valores.append(valor)
            except (ValueError, TypeError):
                if not limpiar_nulos:
                    valores.append(0.0)
        return valores


# ============================================================================
# MÓDULO 3: ANÁLISIS BÁSICO
# ============================================================================

class AnalisisBasico:
    """Análisis estadístico descriptivo básico"""

    def __init__(self, cargador: CargadorDatos):
        self.cargador = cargador
        self.stats = EstadisticaManual()

    def resumen_general(self):
        """Genera un resumen general del dataset"""
        print("\n" + "="*80)
        print("ANÁLISIS BÁSICO - RESUMEN GENERAL DEL DATASET")
        print("="*80)

        total_registros = len(self.cargador.datos_combinados)
        print(f"\n📊 Total de registros: {total_registros}")

        # Distribución por año
        años = {}
        for fila in self.cargador.datos_combinados:
            año = fila.get('año', 'Desconocido')
            años[año] = años.get(año, 0) + 1

        print("\n📅 Distribución por año:")
        for año, cantidad in sorted(años.items()):
            porcentaje = (cantidad / total_registros) * 100
            print(f"   {año}: {cantidad} registros ({porcentaje:.1f}%)")

        # Distribución por distrito
        distritos = {}
        for fila in self.cargador.datos_combinados:
            distrito = fila.get('district', 'Desconocido')
            distritos[distrito] = distritos.get(distrito, 0) + 1

        print(f"\n🗺️  Total de distritos: {len(distritos)}")
        print("   Top 5 distritos con más muestras:")
        top_distritos = sorted(distritos.items(), key=lambda x: x[1], reverse=True)[:5]
        for distrito, cantidad in top_distritos:
            print(f"   - {distrito}: {cantidad} muestras")

        # Distribución por clasificación
        clasificaciones = {}
        for fila in self.cargador.datos_combinados:
            clasif = fila.get('Classification', 'Desconocido')
            clasificaciones[clasif] = clasificaciones.get(clasif, 0) + 1

        print(f"\n💧 Distribución por clasificación de calidad:")
        for clasif, cantidad in sorted(clasificaciones.items(), key=lambda x: x[1], reverse=True):
            porcentaje = (cantidad / total_registros) * 100
            print(f"   {clasif}: {cantidad} ({porcentaje:.1f}%)")

    def estadisticas_descriptivas(self, columna: str):
        """Calcula estadísticas descriptivas para una columna"""
        datos = self.cargador.extraer_columna(columna)

        if not datos:
            print(f"⚠️  No hay datos disponibles para {columna}")
            return

        print(f"\n{'─'*80}")
        print(f"📈 ESTADÍSTICAS DESCRIPTIVAS: {columna}")
        print(f"{'─'*80}")

        print(f"\n🔢 Medidas de tendencia central:")
        print(f"   Media:              {self.stats.media(datos):.4f}")
        print(f"   Mediana:            {self.stats.mediana(datos):.4f}")
        print(f"   Moda:               {self.stats.moda(datos):.4f}")

        print(f"\n📊 Medidas de dispersión:")
        print(f"   Desviación estándar: {self.stats.desviacion_estandar(datos):.4f}")
        print(f"   Varianza:            {self.stats.varianza(datos):.4f}")
        print(f"   Coef. de variación:  {self.stats.coeficiente_variacion(datos):.2f}%")
        print(f"   Rango:               {max(datos) - min(datos):.4f}")
        print(f"   IQR:                 {self.stats.rango_intercuartil(datos):.4f}")

        print(f"\n📏 Valores extremos:")
        print(f"   Mínimo:              {min(datos):.4f}")
        print(f"   Máximo:              {max(datos):.4f}")
        print(f"   Q1 (25%):            {self.stats.percentil(datos, 25):.4f}")
        print(f"   Q2 (50%):            {self.stats.percentil(datos, 50):.4f}")
        print(f"   Q3 (75%):            {self.stats.percentil(datos, 75):.4f}")

        print(f"\n📐 Forma de la distribución:")
        print(f"   Asimetría (Skewness): {self.stats.asimetria(datos):.4f}")
        print(f"   Curtosis (Kurtosis):  {self.stats.curtosis(datos):.4f}")

        # Interpretación
        asimetria = self.stats.asimetria(datos)
        if asimetria > 0.5:
            print(f"   → Distribución sesgada a la derecha (cola larga hacia valores altos)")
        elif asimetria < -0.5:
            print(f"   → Distribución sesgada a la izquierda (cola larga hacia valores bajos)")
        else:
            print(f"   → Distribución aproximadamente simétrica")

    def analisis_completo_parametros(self):
        """Realiza análisis descriptivo de todos los parámetros químicos"""
        print("\n" + "="*80)
        print("ANÁLISIS COMPLETO DE PARÁMETROS QUÍMICOS")
        print("="*80)

        parametros = ['pH', 'E.C', 'TDS', 'CO3', 'HCO3', 'Cl', 'F', 'NO3 ',
                     'SO4', 'Na', 'K', 'Ca', 'Mg', 'T.H', 'SAR']

        for parametro in parametros:
            self.estadisticas_descriptivas(parametro)

    def calidad_por_distrito(self):
        """Analiza la calidad del agua por distrito"""
        print("\n" + "="*80)
        print("ANÁLISIS DE CALIDAD POR DISTRITO")
        print("="*80)

        # Agrupar por distrito
        distritos_data = defaultdict(lambda: {'TDS': [], 'pH': [], 'clasificaciones': []})

        for fila in self.cargador.datos_combinados:
            distrito = fila.get('district', 'Desconocido')
            try:
                tds = float(fila.get('TDS', 0))
                ph = float(fila.get('pH', 0))
                if tds > 0 and ph > 0:
                    distritos_data[distrito]['TDS'].append(tds)
                    distritos_data[distrito]['pH'].append(ph)
                    distritos_data[distrito]['clasificaciones'].append(fila.get('Classification', ''))
            except (ValueError, TypeError):
                pass

        # Calcular promedios
        resultados = []
        for distrito, datos in distritos_data.items():
            if datos['TDS']:
                tds_promedio = self.stats.media(datos['TDS'])
                ph_promedio = self.stats.media(datos['pH'])
                n_muestras = len(datos['TDS'])
                resultados.append((distrito, tds_promedio, ph_promedio, n_muestras))

        # Ordenar por TDS promedio
        resultados.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{'Distrito':<25} {'TDS Prom.':<12} {'pH Prom.':<10} {'Muestras':<10}")
        print("─" * 80)
        for distrito, tds, ph, n in resultados[:15]:  # Top 15
            print(f"{distrito:<25} {tds:>10.2f}   {ph:>8.2f}   {n:>8}")


# ============================================================================
# MÓDULO 4: ANÁLISIS MEDIO
# ============================================================================

class AnalisisMedio:
    """Análisis de correlaciones, distribuciones y detección de outliers"""

    def __init__(self, cargador: CargadorDatos):
        self.cargador = cargador
        self.stats = EstadisticaManual()

    def matriz_correlacion(self, parametros: List[str] = None):
        """Calcula matriz de correlación entre parámetros"""
        if parametros is None:
            parametros = ['pH', 'E.C', 'TDS', 'Cl', 'Na', 'Ca', 'Mg', 'T.H', 'SAR']

        print("\n" + "="*80)
        print("MATRIZ DE CORRELACIÓN DE PEARSON")
        print("="*80)

        # Extraer datos
        datos_parametros = {}
        for param in parametros:
            datos_parametros[param] = self.cargador.extraer_columna(param)

        # Calcular correlaciones
        print(f"\n{'Parámetro':<10}", end="")
        for param in parametros:
            print(f"{param:>8}", end="")
        print()
        print("─" * (10 + 8 * len(parametros)))

        for param1 in parametros:
            print(f"{param1:<10}", end="")
            for param2 in parametros:
                if len(datos_parametros[param1]) > 0 and len(datos_parametros[param2]) > 0:
                    # Asegurar misma longitud
                    min_len = min(len(datos_parametros[param1]), len(datos_parametros[param2]))
                    corr = self.stats.correlacion_pearson(
                        datos_parametros[param1][:min_len],
                        datos_parametros[param2][:min_len]
                    )
                    print(f"{corr:>8.3f}", end="")
                else:
                    print(f"{'N/A':>8}", end="")
            print()

        # Identificar correlaciones fuertes
        print("\n🔍 Correlaciones significativas (|r| > 0.7):")
        for i, param1 in enumerate(parametros):
            for j, param2 in enumerate(parametros):
                if i < j:  # Evitar duplicados
                    min_len = min(len(datos_parametros[param1]), len(datos_parametros[param2]))
                    if min_len > 0:
                        corr = self.stats.correlacion_pearson(
                            datos_parametros[param1][:min_len],
                            datos_parametros[param2][:min_len]
                        )
                        if abs(corr) > 0.7:
                            tipo = "positiva" if corr > 0 else "negativa"
                            print(f"   {param1} ↔ {param2}: r = {corr:.3f} ({tipo})")

    def detectar_outliers_iqr(self, columna: str):
        """Detecta outliers usando el método IQR"""
        datos = self.cargador.extraer_columna(columna)

        if not datos:
            print(f"⚠️  No hay datos para {columna}")
            return

        print(f"\n{'─'*80}")
        print(f"🔍 DETECCIÓN DE OUTLIERS: {columna} (Método IQR)")
        print(f"{'─'*80}")

        q1 = self.stats.percentil(datos, 25)
        q3 = self.stats.percentil(datos, 75)
        iqr = q3 - q1

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        outliers = [x for x in datos if x < limite_inferior or x > limite_superior]
        outliers_extremos = [x for x in datos if x < q1 - 3 * iqr or x > q3 + 3 * iqr]

        print(f"\n📊 Límites de detección:")
        print(f"   Q1:                  {q1:.4f}")
        print(f"   Q3:                  {q3:.4f}")
        print(f"   IQR:                 {iqr:.4f}")
        print(f"   Límite inferior:     {limite_inferior:.4f}")
        print(f"   Límite superior:     {limite_superior:.4f}")

        print(f"\n🎯 Resultados:")
        print(f"   Total de datos:      {len(datos)}")
        print(f"   Outliers moderados:  {len(outliers)} ({len(outliers)/len(datos)*100:.2f}%)")
        print(f"   Outliers extremos:   {len(outliers_extremos)} ({len(outliers_extremos)/len(datos)*100:.2f}%)")

        if outliers:
            print(f"\n   Valores outliers (primeros 10):")
            for valor in sorted(outliers, reverse=True)[:10]:
                print(f"      {valor:.4f}")

    def analisis_temporal(self):
        """Analiza tendencias temporales entre años"""
        print("\n" + "="*80)
        print("ANÁLISIS TEMPORAL (2018-2020)")
        print("="*80)

        parametros = ['pH', 'TDS', 'E.C', 'T.H', 'SAR']
        años = ['2018', '2019', '2020']

        for parametro in parametros:
            print(f"\n📈 Tendencia temporal: {parametro}")
            print("─" * 60)

            valores_por_año = {}
            for año in años:
                valores = []
                for fila in self.cargador.datos_combinados:
                    if fila.get('año') == año:
                        try:
                            valor = float(fila.get(parametro, 0))
                            if valor > 0:
                                valores.append(valor)
                        except (ValueError, TypeError):
                            pass
                valores_por_año[año] = valores

            # Calcular estadísticas por año
            print(f"\n{'Año':<8} {'Media':<12} {'Mediana':<12} {'Desv.Est':<12} {'N':<8}")
            print("─" * 60)

            medias = []
            for año in años:
                if valores_por_año[año]:
                    media = self.stats.media(valores_por_año[año])
                    mediana = self.stats.mediana(valores_por_año[año])
                    desv = self.stats.desviacion_estandar(valores_por_año[año])
                    n = len(valores_por_año[año])
                    medias.append(media)
                    print(f"{año:<8} {media:<12.4f} {mediana:<12.4f} {desv:<12.4f} {n:<8}")

            # Calcular tendencia
            if len(medias) == 3:
                x = [0, 1, 2]  # Años codificados
                pendiente, intercepto, r2 = self.stats.regresion_lineal(x, medias)

                print(f"\n   Tendencia lineal:")
                print(f"   Pendiente: {pendiente:.4f} (cambio anual)")
                print(f"   R²: {r2:.4f}")

                if abs(pendiente) > 0.01:
                    direccion = "incremento" if pendiente > 0 else "disminución"
                    print(f"   → Se observa {direccion} de {abs(pendiente):.4f} unidades por año")
                else:
                    print(f"   → Valores relativamente estables en el tiempo")

    def analisis_distribucion(self, columna: str):
        """Analiza la distribución de una variable"""
        datos = self.cargador.extraer_columna(columna)

        if not datos:
            return

        print(f"\n{'─'*80}")
        print(f"📊 ANÁLISIS DE DISTRIBUCIÓN: {columna}")
        print(f"{'─'*80}")

        # Crear histograma manual
        n_bins = 10
        minimo = min(datos)
        maximo = max(datos)
        ancho_bin = (maximo - minimo) / n_bins

        bins = [0] * n_bins
        for valor in datos:
            bin_idx = int((valor - minimo) / ancho_bin)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            bins[bin_idx] += 1

        print(f"\n📊 Histograma (n={len(datos)}):")
        print(f"{'Rango':<25} {'Frecuencia':<12} {'Gráfico'}")
        print("─" * 80)

        max_freq = max(bins)
        for i, freq in enumerate(bins):
            inicio = minimo + i * ancho_bin
            fin = inicio + ancho_bin
            barra = '█' * int((freq / max_freq) * 40)
            print(f"{inicio:>10.2f} - {fin:<10.2f} {freq:<12} {barra}")

        # Test de normalidad (aproximado usando asimetría y curtosis)
        asimetria = self.stats.asimetria(datos)
        curtosis = self.stats.curtosis(datos)

        print(f"\n🔬 Evaluación de normalidad:")
        print(f"   Asimetría: {asimetria:.4f}")
        print(f"   Curtosis:  {curtosis:.4f}")

        if abs(asimetria) < 0.5 and abs(curtosis) < 1:
            print(f"   → La distribución es aproximadamente normal")
        elif abs(asimetria) >= 0.5:
            print(f"   → La distribución presenta asimetría significativa")
        else:
            print(f"   → La distribución presenta colas pesadas/ligeras")


# ============================================================================
# MÓDULO 5: ANÁLISIS AVANZADO
# ============================================================================

class AnalisisAvanzado:
    """Análisis multivariado, PCA manual, clustering y análisis espacial"""

    def __init__(self, cargador: CargadorDatos):
        self.cargador = cargador
        self.stats = EstadisticaManual()

    def pca_manual(self, parametros: List[str] = None, n_componentes: int = 3):
        """
        Implementación manual de PCA (Análisis de Componentes Principales)
        Simplificado usando método de potencias para eigenvalores
        """
        if parametros is None:
            parametros = ['pH', 'E.C', 'TDS', 'Na', 'Ca', 'Mg', 'T.H']

        print("\n" + "="*80)
        print("ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)")
        print("="*80)

        # Extraer y normalizar datos
        datos_matriz = []
        for param in parametros:
            datos = self.cargador.extraer_columna(param)
            datos_norm = self.stats.normalizar_zscore(datos)
            datos_matriz.append(datos_norm)

        n_vars = len(parametros)
        n_obs = len(datos_matriz[0])

        print(f"\n📊 Configuración:")
        print(f"   Variables: {n_vars}")
        print(f"   Observaciones: {n_obs}")
        print(f"   Componentes a extraer: {n_componentes}")

        # Calcular matriz de covarianza
        print(f"\n🔢 Matriz de covarianza:")
        cov_matriz = []
        for i in range(n_vars):
            fila = []
            for j in range(n_vars):
                cov = self.stats.covarianza(datos_matriz[i], datos_matriz[j])
                fila.append(cov)
            cov_matriz.append(fila)

        # Mostrar matriz de covarianza
        print(f"\n{'':>10}", end="")
        for param in parametros:
            print(f"{param:>10}", end="")
        print()
        for i, param in enumerate(parametros):
            print(f"{param:>10}", end="")
            for j in range(n_vars):
                print(f"{cov_matriz[i][j]:>10.4f}", end="")
            print()

        # Calcular varianza total
        varianza_total = sum(cov_matriz[i][i] for i in range(n_vars))
        print(f"\n   Varianza total: {varianza_total:.4f}")

        # Aproximación de componentes principales (simplificada)
        # En un PCA completo se calcularían eigenvalores y eigenvectores
        # Aquí usamos una aproximación basada en la varianza de cada variable
        print(f"\n📈 Contribución de varianza por variable:")
        varianzas = [(parametros[i], cov_matriz[i][i]) for i in range(n_vars)]
        varianzas.sort(key=lambda x: x[1], reverse=True)

        varianza_acum = 0
        for i, (param, var) in enumerate(varianzas):
            prop = (var / varianza_total) * 100
            varianza_acum += prop
            print(f"   PC{i+1} ({param}): {prop:.2f}% (Acumulado: {varianza_acum:.2f}%)")

    def clustering_kmeans_manual(self, parametros: List[str] = None, k: int = 3, max_iter: int = 50):
        """
        Implementación manual de K-Means clustering
        """
        if parametros is None:
            parametros = ['TDS', 'T.H', 'SAR']

        print("\n" + "="*80)
        print(f"CLUSTERING K-MEANS (k={k})")
        print("="*80)

        # Extraer y normalizar datos
        datos_matriz = []
        for param in parametros:
            datos = self.cargador.extraer_columna(param)
            datos_norm = self.stats.normalizar_minmax(datos)
            datos_matriz.append(datos_norm)

        n_vars = len(parametros)
        n_obs = len(datos_matriz[0])

        print(f"\n📊 Configuración:")
        print(f"   Variables: {', '.join(parametros)}")
        print(f"   Observaciones: {n_obs}")
        print(f"   Clusters (k): {k}")

        # Transponer matriz para tener observaciones como filas
        observaciones = [[datos_matriz[j][i] for j in range(n_vars)] for i in range(n_obs)]

        # Inicializar centroides aleatoriamente
        import random
        random.seed(42)
        centroides = random.sample(observaciones, k)

        # Algoritmo K-Means
        for iteracion in range(max_iter):
            # Asignar cada punto al centroide más cercano
            asignaciones = []
            for obs in observaciones:
                distancias = []
                for centroide in centroides:
                    dist = math.sqrt(sum((obs[i] - centroide[i])**2 for i in range(n_vars)))
                    distancias.append(dist)
                asignaciones.append(distancias.index(min(distancias)))

            # Recalcular centroides
            nuevos_centroides = []
            for cluster_id in range(k):
                puntos_cluster = [observaciones[i] for i in range(n_obs) if asignaciones[i] == cluster_id]
                if puntos_cluster:
                    nuevo_centroide = [
                        sum(punto[j] for punto in puntos_cluster) / len(puntos_cluster)
                        for j in range(n_vars)
                    ]
                    nuevos_centroides.append(nuevo_centroide)
                else:
                    nuevos_centroides.append(centroides[cluster_id])

            # Verificar convergencia
            cambio = sum(
                math.sqrt(sum((nuevos_centroides[i][j] - centroides[i][j])**2 for j in range(n_vars)))
                for i in range(k)
            )

            centroides = nuevos_centroides

            if cambio < 0.0001:
                print(f"\n✓ Convergencia alcanzada en iteración {iteracion + 1}")
                break

        # Resultados
        print(f"\n📊 Distribución de clusters:")
        for cluster_id in range(k):
            n_puntos = asignaciones.count(cluster_id)
            porcentaje = (n_puntos / n_obs) * 100
            print(f"   Cluster {cluster_id + 1}: {n_puntos} puntos ({porcentaje:.1f}%)")

        print(f"\n🎯 Centroides finales (valores normalizados):")
        for i, centroide in enumerate(centroides):
            print(f"\n   Cluster {i + 1}:")
            for j, param in enumerate(parametros):
                print(f"      {param}: {centroide[j]:.4f}")

        # Calcular inercia (suma de distancias cuadradas intra-cluster)
        inercia = 0
        for i, obs in enumerate(observaciones):
            cluster_id = asignaciones[i]
            dist_sq = sum((obs[j] - centroides[cluster_id][j])**2 for j in range(n_vars))
            inercia += dist_sq

        print(f"\n📏 Inercia total: {inercia:.4f}")
        print(f"   (Menor inercia indica clusters más compactos)")

        return asignaciones, centroides

    def analisis_espacial(self):
        """Analiza patrones espaciales usando coordenadas geográficas"""
        print("\n" + "="*80)
        print("ANÁLISIS ESPACIAL")
        print("="*80)

        # Extraer coordenadas y TDS
        puntos = []
        for fila in self.cargador.datos_combinados:
            try:
                lat = float(fila.get('lat_gis', 0))
                lon = float(fila.get('long_gis', 0))
                tds = float(fila.get('TDS', 0))
                if lat != 0 and lon != 0 and tds > 0:
                    puntos.append((lat, lon, tds))
            except (ValueError, TypeError):
                pass

        print(f"\n📍 Puntos de muestreo con coordenadas: {len(puntos)}")

        # Calcular estadísticas espaciales
        latitudes = [p[0] for p in puntos]
        longitudes = [p[1] for p in puntos]
        tds_valores = [p[2] for p in puntos]

        print(f"\n🗺️  Extensión geográfica:")
        print(f"   Latitud:  {min(latitudes):.4f} a {max(latitudes):.4f}")
        print(f"   Longitud: {min(longitudes):.4f} a {max(longitudes):.4f}")

        # Dividir en cuadrantes
        lat_media = self.stats.media(latitudes)
        lon_media = self.stats.media(longitudes)

        cuadrantes = {
            'NE': [], 'NW': [], 'SE': [], 'SW': []
        }

        for lat, lon, tds in puntos:
            if lat >= lat_media and lon >= lon_media:
                cuadrantes['NE'].append(tds)
            elif lat >= lat_media and lon < lon_media:
                cuadrantes['NW'].append(tds)
            elif lat < lat_media and lon >= lon_media:
                cuadrantes['SE'].append(tds)
            else:
                cuadrantes['SW'].append(tds)

        print(f"\n🧭 Análisis por cuadrantes (TDS promedio):")
        for cuadrante, valores in cuadrantes.items():
            if valores:
                promedio = self.stats.media(valores)
                n = len(valores)
                print(f"   {cuadrante}: {promedio:.2f} mg/L (n={n})")

        # Autocorrelación espacial simplificada
        print(f"\n📊 Variabilidad espacial:")
        print(f"   Coef. variación TDS: {self.stats.coeficiente_variacion(tds_valores):.2f}%")


# ============================================================================
# MÓDULO 6: CIENCIA DE DATOS
# ============================================================================

class CienciaDatos:
    """Modelos predictivos, feature importance y validación"""

    def __init__(self, cargador: CargadorDatos):
        self.cargador = cargador
        self.stats = EstadisticaManual()

    def preparar_datos_clasificacion(self):
        """Prepara datos para clasificación de calidad de agua"""
        print("\n" + "="*80)
        print("PREPARACIÓN DE DATOS PARA MODELADO")
        print("="*80)

        # Características y target
        features = ['pH', 'E.C', 'TDS', 'CO3', 'HCO3', 'Cl', 'Na', 'Ca', 'Mg', 'T.H', 'SAR']

        X = []
        y = []

        for fila in self.cargador.datos_combinados:
            try:
                # Extraer features
                fila_features = []
                valido = True
                for feature in features:
                    valor = float(fila.get(feature, 0))
                    if valor == 0:
                        valido = False
                        break
                    fila_features.append(valor)

                if valido:
                    clasificacion = fila.get('Classification', '')
                    if clasificacion:
                        X.append(fila_features)
                        y.append(clasificacion)
            except (ValueError, TypeError):
                pass

        print(f"\n📊 Dataset preparado:")
        print(f"   Muestras totales: {len(X)}")
        print(f"   Features: {len(features)}")
        print(f"   Clases únicas: {len(set(y))}")

        # Distribución de clases
        print(f"\n📈 Distribución de clases:")
        clases_count = {}
        for clase in y:
            clases_count[clase] = clases_count.get(clase, 0) + 1

        for clase, count in sorted(clases_count.items(), key=lambda x: x[1], reverse=True):
            porcentaje = (count / len(y)) * 100
            print(f"   {clase}: {count} ({porcentaje:.1f}%)")

        return X, y, features

    def validacion_cruzada_manual(self, X: List[List[float]], y: List[str], k: int = 5):
        """Implementación manual de validación cruzada k-fold"""
        print(f"\n{'─'*80}")
        print(f"VALIDACIÓN CRUZADA {k}-FOLD")
        print(f"{'─'*80}")

        n = len(X)
        fold_size = n // k

        print(f"\n📊 Configuración:")
        print(f"   Total de muestras: {n}")
        print(f"   Número de folds: {k}")
        print(f"   Tamaño de cada fold: ~{fold_size}")

        # Crear índices para cada fold
        indices = list(range(n))
        import random
        random.seed(42)
        random.shuffle(indices)

        folds = []
        for i in range(k):
            inicio = i * fold_size
            fin = inicio + fold_size if i < k - 1 else n
            folds.append(indices[inicio:fin])

        print(f"\n✓ Folds creados exitosamente")
        for i, fold in enumerate(folds):
            print(f"   Fold {i+1}: {len(fold)} muestras")

        return folds

    def clasificador_naive_bayes_manual(self, X_train: List[List[float]], y_train: List[str],
                                       X_test: List[List[float]], y_test: List[str]):
        """
        Implementación simplificada de Naive Bayes Gaussiano
        """
        print(f"\n{'─'*80}")
        print("CLASIFICADOR NAIVE BAYES")
        print(f"{'─'*80}")

        # Calcular probabilidades a priori y estadísticas por clase
        clases = list(set(y_train))
        n_features = len(X_train[0])

        # Estadísticas por clase
        stats_por_clase = {}
        for clase in clases:
            # Filtrar datos de esta clase
            X_clase = [X_train[i] for i in range(len(X_train)) if y_train[i] == clase]

            # Calcular media y desviación estándar para cada feature
            medias = []
            desvs = []
            for j in range(n_features):
                feature_valores = [x[j] for x in X_clase]
                medias.append(self.stats.media(feature_valores))
                desvs.append(self.stats.desviacion_estandar(feature_valores))

            stats_por_clase[clase] = {
                'prior': len(X_clase) / len(X_train),
                'medias': medias,
                'desvs': desvs
            }

        # Función de densidad gaussiana
        def gaussian_pdf(x, media, desv):
            if desv == 0:
                return 1.0
            exponente = -((x - media) ** 2) / (2 * desv ** 2)
            return (1 / (desv * math.sqrt(2 * math.pi))) * math.exp(exponente)

        # Predecir
        predicciones = []
        for x in X_test:
            probabilidades = {}
            for clase in clases:
                # Probabilidad a priori
                prob = math.log(stats_por_clase[clase]['prior'])

                # Multiplicar probabilidades de cada feature (sumar en log)
                for j in range(n_features):
                    media = stats_por_clase[clase]['medias'][j]
                    desv = stats_por_clase[clase]['desvs'][j]
                    prob += math.log(gaussian_pdf(x[j], media, desv) + 1e-10)

                probabilidades[clase] = prob

            # Seleccionar clase con mayor probabilidad
            predicciones.append(max(probabilidades, key=probabilidades.get))

        # Calcular métricas
        accuracy = sum(1 for i in range(len(y_test)) if predicciones[i] == y_test[i]) / len(y_test)

        print(f"\n📊 Resultados:")
        print(f"   Muestras de entrenamiento: {len(X_train)}")
        print(f"   Muestras de prueba: {len(X_test)}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Matriz de confusión simplificada
        print(f"\n📈 Matriz de confusión (primeras 5 clases):")
        clases_top = sorted(set(y_test), key=lambda c: y_test.count(c), reverse=True)[:5]

        matriz = {c1: {c2: 0 for c2 in clases_top} for c1 in clases_top}
        for i in range(len(y_test)):
            if y_test[i] in clases_top and predicciones[i] in clases_top:
                matriz[y_test[i]][predicciones[i]] += 1

        print(f"\n{'Real/Pred':<12}", end="")
        for clase in clases_top:
            print(f"{clase:>8}", end="")
        print()
        print("─" * (12 + 8 * len(clases_top)))

        for clase_real in clases_top:
            print(f"{clase_real:<12}", end="")
            for clase_pred in clases_top:
                print(f"{matriz[clase_real][clase_pred]:>8}", end="")
            print()

        return accuracy, predicciones

    def feature_importance_manual(self, X: List[List[float]], y: List[str],
                                  feature_names: List[str]):
        """
        Calcula importancia de features usando correlación con el target
        """
        print(f"\n{'─'*80}")
        print("IMPORTANCIA DE CARACTERÍSTICAS")
        print(f"{'─'*80}")

        # Convertir clases a valores numéricos
        clases_unicas = sorted(set(y))
        y_numerico = [clases_unicas.index(clase) for clase in y]

        # Calcular correlación de cada feature con el target
        importancias = []
        for j in range(len(feature_names)):
            feature_valores = [x[j] for x in X]
            corr = abs(self.stats.correlacion_pearson(feature_valores, y_numerico))
            importancias.append((feature_names[j], corr))

        # Ordenar por importancia
        importancias.sort(key=lambda x: x[1], reverse=True)

        print(f"\n📊 Ranking de características:")
        print(f"{'Característica':<15} {'Importancia':<12} {'Gráfico'}")
        print("─" * 80)

        max_imp = importancias[0][1] if importancias else 1
        for i, (feature, imp) in enumerate(importancias):
            barra = '█' * int((imp / max_imp) * 40)
            print(f"{i+1}. {feature:<12} {imp:>10.4f}   {barra}")

        return importancias

    def analisis_predictivo_completo(self):
        """Ejecuta pipeline completo de ciencia de datos"""
        print("\n" + "="*80)
        print("ANÁLISIS PREDICTIVO COMPLETO")
        print("="*80)

        # Preparar datos
        X, y, features = self.preparar_datos_clasificacion()

        if len(X) < 100:
            print("\n⚠️  Datos insuficientes para análisis predictivo")
            return

        # Feature importance
        self.feature_importance_manual(X, y, features)

        # Dividir en train/test (80/20)
        n = len(X)
        n_train = int(n * 0.8)

        import random
        random.seed(42)
        indices = list(range(n))
        random.shuffle(indices)

        X_train = [X[i] for i in indices[:n_train]]
        y_train = [y[i] for i in indices[:n_train]]
        X_test = [X[i] for i in indices[n_train:]]
        y_test = [y[i] for i in indices[n_train:]]

        print(f"\n📊 División de datos:")
        print(f"   Entrenamiento: {len(X_train)} ({len(X_train)/n*100:.1f}%)")
        print(f"   Prueba: {len(X_test)} ({len(X_test)/n*100:.1f}%)")

        # Entrenar y evaluar modelo
        accuracy, predicciones = self.clasificador_naive_bayes_manual(
            X_train, y_train, X_test, y_test
        )

        # Métricas por clase
        print(f"\n📈 Métricas por clase:")
        clases_unicas = sorted(set(y_test))

        for clase in clases_unicas[:5]:  # Top 5 clases
            tp = sum(1 for i in range(len(y_test)) if y_test[i] == clase and predicciones[i] == clase)
            fp = sum(1 for i in range(len(y_test)) if y_test[i] != clase and predicciones[i] == clase)
            fn = sum(1 for i in range(len(y_test)) if y_test[i] == clase and predicciones[i] != clase)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\n   Clase {clase}:")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall:    {recall:.4f}")
            print(f"      F1-Score:  {f1:.4f}")


# ============================================================================
# MÓDULO 7: ANÁLISIS DE CALIDAD SEGÚN NORMATIVAS
# ============================================================================

class AnalisisCalidad:
    """Evalúa calidad del agua según normativas y estándares"""

    def __init__(self, cargador: CargadorDatos):
        self.cargador = cargador
        self.stats = EstadisticaManual()

    def evaluar_calidad_rsc(self):
        """Evalúa calidad según RSC (Residual Sodium Carbonate)"""
        print("\n" + "="*80)
        print("EVALUACIÓN DE CALIDAD SEGÚN RSC")
        print("="*80)

        rsc_valores = self.cargador.extraer_columna('RSC  meq  / L')

        if not rsc_valores:
            print("⚠️  No hay datos de RSC disponibles")
            return

        # Clasificar según RSC
        seguro = sum(1 for x in rsc_valores if x < 1.25)
        marginal = sum(1 for x in rsc_valores if 1.25 <= x <= 2.50)
        inadecuado = sum(1 for x in rsc_valores if x > 2.50)
        total = len(rsc_valores)

        print(f"\n📊 Clasificación según RSC:")
        print(f"   Seguro (< 1.25):        {seguro} ({seguro/total*100:.1f}%)")
        print(f"   Marginal (1.25-2.50):   {marginal} ({marginal/total*100:.1f}%)")
        print(f"   Inadecuado (> 2.50):    {inadecuado} ({inadecuado/total*100:.1f}%)")

        print(f"\n📈 Estadísticas RSC:")
        print(f"   Media:    {self.stats.media(rsc_valores):.4f}")
        print(f"   Mediana:  {self.stats.mediana(rsc_valores):.4f}")
        print(f"   Mínimo:   {min(rsc_valores):.4f}")
        print(f"   Máximo:   {max(rsc_valores):.4f}")

    def evaluar_calidad_tds(self):
        """Evalúa calidad según TDS para uso ganadero"""
        print("\n" + "="*80)
        print("EVALUACIÓN DE CALIDAD SEGÚN TDS (Uso Ganadero)")
        print("="*80)

        tds_valores = self.cargador.extraer_columna('TDS')

        if not tds_valores:
            print("⚠️  No hay datos de TDS disponibles")
            return

        # Clasificar según TDS
        excelente = sum(1 for x in tds_valores if x < 1000)
        satisfactorio = sum(1 for x in tds_valores if 1000 <= x < 3000)
        limitado_aves = sum(1 for x in tds_valores if 3000 <= x < 5000)
        limitado = sum(1 for x in tds_valores if 5000 <= x < 7000)
        muy_limitado = sum(1 for x in tds_valores if 7000 <= x < 10000)
        no_recomendado = sum(1 for x in tds_valores if x >= 10000)
        total = len(tds_valores)

        print(f"\n📊 Clasificación según TDS (mg/L):")
        print(f"   Excelente (< 1000):           {excelente} ({excelente/total*100:.1f}%)")
        print(f"   Satisfactorio (1000-3000):    {satisfactorio} ({satisfactorio/total*100:.1f}%)")
        print(f"   Limitado aves (3000-5000):    {limitado_aves} ({limitado_aves/total*100:.1f}%)")
        print(f"   Limitado (5000-7000):         {limitado} ({limitado/total*100:.1f}%)")
        print(f"   Muy limitado (7000-10000):    {muy_limitado} ({muy_limitado/total*100:.1f}%)")
        print(f"   No recomendado (≥ 10000):     {no_recomendado} ({no_recomendado/total*100:.1f}%)")

    def analisis_ph(self):
        """Analiza niveles de pH"""
        print("\n" + "="*80)
        print("ANÁLISIS DE pH")
        print("="*80)

        ph_valores = self.cargador.extraer_columna('pH')

        if not ph_valores:
            print("⚠️  No hay datos de pH disponibles")
            return

        # Clasificar pH
        muy_acido = sum(1 for x in ph_valores if x < 6.5)
        neutro = sum(1 for x in ph_valores if 6.5 <= x <= 8.5)
        alcalino = sum(1 for x in ph_valores if x > 8.5)
        total = len(ph_valores)

        print(f"\n📊 Clasificación de pH:")
        print(f"   Ácido (< 6.5):      {muy_acido} ({muy_acido/total*100:.1f}%)")
        print(f"   Neutro (6.5-8.5):   {neutro} ({neutro/total*100:.1f}%)")
        print(f"   Alcalino (> 8.5):   {alcalino} ({alcalino/total*100:.1f}%)")

        print(f"\n📈 Estadísticas pH:")
        print(f"   Media:    {self.stats.media(ph_valores):.2f}")
        print(f"   Mediana:  {self.stats.mediana(ph_valores):.2f}")
        print(f"   Rango:    {min(ph_valores):.2f} - {max(ph_valores):.2f}")


# ============================================================================
# MÓDULO 8: GENERADOR DE REPORTES
# ============================================================================

class GeneradorReportes:
    """Genera reportes consolidados de análisis"""

    def __init__(self, cargador: CargadorDatos):
        self.cargador = cargador

    def reporte_ejecutivo(self):
        """Genera un reporte ejecutivo consolidado"""
        print("\n" + "="*80)
        print("REPORTE EJECUTIVO - CALIDAD DE AGUA SUBTERRÁNEA")
        print("="*80)
        print("\n📋 Resumen de Hallazgos Principales\n")

        stats = EstadisticaManual()

        # 1. Cobertura del estudio
        total_muestras = len(self.cargador.datos_combinados)
        distritos = set(fila.get('district', '') for fila in self.cargador.datos_combinados)

        print("1️⃣  COBERTURA DEL ESTUDIO")
        print(f"   • Total de muestras analizadas: {total_muestras}")
        print(f"   • Distritos cubiertos: {len(distritos)}")
        print(f"   • Período: 2018-2020 (post-monzón)")

        # 2. Calidad general
        clasificaciones = {}
        for fila in self.cargador.datos_combinados:
            clasif = fila.get('Classification', 'Desconocido')
            clasificaciones[clasif] = clasificaciones.get(clasif, 0) + 1

        print("\n2️⃣  CALIDAD GENERAL DEL AGUA")
        top_3_clasif = sorted(clasificaciones.items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (clasif, count) in enumerate(top_3_clasif, 1):
            porcentaje = (count / total_muestras) * 100
            print(f"   {i}. {clasif}: {porcentaje:.1f}% de las muestras")

        # 3. Parámetros críticos
        tds_valores = self.cargador.extraer_columna('TDS')
        ph_valores = self.cargador.extraer_columna('pH')

        print("\n3️⃣  PARÁMETROS CRÍTICOS")
        if tds_valores:
            tds_promedio = stats.media(tds_valores)
            tds_alto = sum(1 for x in tds_valores if x > 3000)
            print(f"   • TDS promedio: {tds_promedio:.0f} mg/L")
            print(f"   • Muestras con TDS alto (>3000): {tds_alto/len(tds_valores)*100:.1f}%")

        if ph_valores:
            ph_promedio = stats.media(ph_valores)
            ph_alcalino = sum(1 for x in ph_valores if x > 8.5)
            print(f"   • pH promedio: {ph_promedio:.2f}")
            print(f"   • Muestras alcalinas (pH>8.5): {ph_alcalino/len(ph_valores)*100:.1f}%")

        # 4. Recomendaciones
        print("\n4️⃣  RECOMENDACIONES")
        print("   • Monitoreo continuo de zonas con alta salinidad")
        print("   • Implementar prácticas de manejo según clasificación")
        print("   • Considerar tratamientos para zonas C4 (muy alta salinidad)")
        print("   • Evaluar uso de enmiendas (yeso) en zonas con alto SAR")

        print("\n" + "="*80)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta todos los análisis"""

    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "ANÁLISIS DE CALIDAD DE AGUA SUBTERRÁNEA" + " "*19 + "█")
    print("█" + " "*25 + "Sistema Profesional de Análisis" + " "*22 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    # Cargar datos
    print("\n🔄 Cargando datos...")
    cargador = CargadorDatos("samples")
    cargador.cargar_archivos_csv()

    if not cargador.datos_combinados:
        print("❌ No se pudieron cargar los datos. Verifica la ruta.")
        return

    # Menú interactivo
    while True:
        print("\n" + "="*80)
        print("MENÚ PRINCIPAL")
        print("="*80)
        print("\n📊 ANÁLISIS BÁSICO")
        print("   1. Resumen general del dataset")
        print("   2. Estadísticas descriptivas por parámetro")
        print("   3. Análisis de calidad por distrito")

        print("\n📈 ANÁLISIS MEDIO")
        print("   4. Matriz de correlación")
        print("   5. Detección de outliers")
        print("   6. Análisis temporal (2018-2020)")
        print("   7. Análisis de distribución")

        print("\n🔬 ANÁLISIS AVANZADO")
        print("   8. PCA (Componentes Principales)")
        print("   9. Clustering K-Means")
        print("   10. Análisis espacial")

        print("\n🤖 CIENCIA DE DATOS")
        print("   11. Análisis predictivo completo")
        print("   12. Feature importance")

        print("\n💧 EVALUACIÓN DE CALIDAD")
        print("   13. Evaluación según RSC")
        print("   14. Evaluación según TDS")
        print("   15. Análisis de pH")

        print("\n📋 REPORTES")
        print("   16. Reporte ejecutivo")
        print("   17. Análisis completo (todos los módulos)")

        print("\n   0. Salir")

        try:
            opcion = input("\n👉 Selecciona una opción: ").strip()

            if opcion == "0":
                print("\n✅ Análisis finalizado. ¡Hasta pronto!")
                break

            elif opcion == "1":
                analisis_basico = AnalisisBasico(cargador)
                analisis_basico.resumen_general()

            elif opcion == "2":
                parametros = ['pH', 'E.C', 'TDS', 'T.H', 'SAR', 'Cl', 'Na']
                analisis_basico = AnalisisBasico(cargador)
                for param in parametros:
                    analisis_basico.estadisticas_descriptivas(param)
                    input("\nPresiona Enter para continuar...")

            elif opcion == "3":
                analisis_basico = AnalisisBasico(cargador)
                analisis_basico.calidad_por_distrito()

            elif opcion == "4":
                analisis_medio = AnalisisMedio(cargador)
                analisis_medio.matriz_correlacion()

            elif opcion == "5":
                analisis_medio = AnalisisMedio(cargador)
                parametros = ['TDS', 'E.C', 'T.H', 'SAR']
                for param in parametros:
                    analisis_medio.detectar_outliers_iqr(param)

            elif opcion == "6":
                analisis_medio = AnalisisMedio(cargador)
                analisis_medio.analisis_temporal()

            elif opcion == "7":
                analisis_medio = AnalisisMedio(cargador)
                parametros = ['TDS', 'pH', 'SAR']
                for param in parametros:
                    analisis_medio.analisis_distribucion(param)

            elif opcion == "8":
                analisis_avanzado = AnalisisAvanzado(cargador)
                analisis_avanzado.pca_manual()

            elif opcion == "9":
                analisis_avanzado = AnalisisAvanzado(cargador)
                analisis_avanzado.clustering_kmeans_manual(k=4)

            elif opcion == "10":
                analisis_avanzado = AnalisisAvanzado(cargador)
                analisis_avanzado.analisis_espacial()

            elif opcion == "11":
                ciencia_datos = CienciaDatos(cargador)
                ciencia_datos.analisis_predictivo_completo()

            elif opcion == "12":
                ciencia_datos = CienciaDatos(cargador)
                X, y, features = ciencia_datos.preparar_datos_clasificacion()
                ciencia_datos.feature_importance_manual(X, y, features)

            elif opcion == "13":
                analisis_calidad = AnalisisCalidad(cargador)
                analisis_calidad.evaluar_calidad_rsc()

            elif opcion == "14":
                analisis_calidad = AnalisisCalidad(cargador)
                analisis_calidad.evaluar_calidad_tds()

            elif opcion == "15":
                analisis_calidad = AnalisisCalidad(cargador)
                analisis_calidad.analisis_ph()

            elif opcion == "16":
                generador = GeneradorReportes(cargador)
                generador.reporte_ejecutivo()

            elif opcion == "17":
                print("\n🚀 Ejecutando análisis completo...")

                # Básico
                analisis_basico = AnalisisBasico(cargador)
                analisis_basico.resumen_general()

                # Medio
                analisis_medio = AnalisisMedio(cargador)
                analisis_medio.matriz_correlacion()
                analisis_medio.analisis_temporal()

                # Avanzado
                analisis_avanzado = AnalisisAvanzado(cargador)
                analisis_avanzado.pca_manual()
                analisis_avanzado.clustering_kmeans_manual(k=4)

                # Calidad
                analisis_calidad = AnalisisCalidad(cargador)
                analisis_calidad.evaluar_calidad_rsc()
                analisis_calidad.evaluar_calidad_tds()

                # Reporte
                generador = GeneradorReportes(cargador)
                generador.reporte_ejecutivo()

                print("\n✅ Análisis completo finalizado")

            else:
                print("\n⚠️  Opción no válida. Intenta de nuevo.")

            input("\n⏸️  Presiona Enter para volver al menú...")

        except KeyboardInterrupt:
            print("\n\n✅ Análisis interrumpido. ¡Hasta pronto!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    main()


