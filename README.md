# Proyecto de Análisis de Datos - Clasificación de Hongos

## Contexto del Proyecto

Este es un **proyecto final para el curso de Análisis de Datos** enfocado en la clasificación de hongos usando el **UCI Secondary Mushroom Dataset**. El objetivo principal es **predecir si un hongo es comestible o venenoso** basándose en sus características físicas utilizando técnicas de machine learning.

**⚠️ ADVERTENCIA DE SALUD PÚBLICA:**
Este es un problema crítico donde los errores pueden tener consecuencias fatales:
- **Falso Negativo** (clasificar hongo venenoso como comestible): **INACEPTABLE** - puede causar muerte
- **Falso Positivo** (clasificar hongo comestible como venenoso): **ACEPTABLE** - solo causa rechazo innecesario

---

## 📋 Tabla de Contenidos

1. [Estructura del Proyecto](#estructura-del-proyecto)
2. [Dataset](#dataset)
3. [Flujo de Trabajo](#flujo-de-trabajo)
4. [Resultados del Modelo](#resultados-del-modelo)
5. [Instalación y Ejecución](#instalación-y-ejecución)
6. [Archivos Principales](#archivos-principales)
7. [Problemas de Calidad de Datos](#problemas-de-calidad-de-datos)
8. [Autores](#autores)

---

## 📁 Estructura del Proyecto

```
Proy_AD/
├── README.md                          # Este archivo
├── CLAUDE.md                          # Documentación técnica del proyecto
│
├── MushroomDataset/                   # Datos
│   ├── MushroomDataset.csv           # Dataset original (61,079 filas, delimiter=',')
│   ├── MushroomDataset_cleaned.csv   # Dataset limpio (53,541 filas, delimiter=';')
│   ├── secondary_data.csv            # Copia alternativa del dataset original
│   ├── primary_data.csv              # Datos fuente (173 especies)
│   ├── secondary_data_meta.txt       # Metadatos y codificación de variables
│   └── primary_data_meta.txt         # Metadatos del dataset primario
│
├── AnalisisExploratorio.ipynb        # 1️⃣ Análisis exploratorio de datos
├── Limpieza_Datos.ipynb              # 2️⃣ Proceso de limpieza de datos
├── Analisis_Visualizaciones.ipynb    # 3️⃣ Visualizaciones y patrones
├── Modelo_Predictivo.ipynb           # 4️⃣ Modelo predictivo completo ✅
└── Codigo.ipynb                       # Notebook legacy
```

---

## 🍄 Dataset

### Información General

| Característica | Valor |
|----------------|-------|
| **Fuente** | UCI Machine Learning Repository |
| **Tipo** | Secondary Mushroom Dataset |
| **Filas originales** | 61,079 |
| **Filas limpias** | 53,541 (87.66% retenido) |
| **Columnas** | 21 → 17 (tras limpieza) |
| **Especies** | 173 especies de hongos |
| **Muestras por especie** | 353 hipotéticas |

### Variable Objetivo

- **`class`**: Clasificación del hongo
  - `e` (edible): Comestible - seguro para consumo humano
  - `p` (poisonous): Venenoso - peligroso, puede causar intoxicación

### Variables Predictoras

**Variables Numéricas (3):**
- `cap-diameter`: Diámetro del sombrero (cm)
- `stem-height`: Altura del tallo (cm)
- `stem-width`: Ancho del tallo (mm)

**Variables Categóricas (13):**
- Características del sombrero: `cap-shape`, `cap-surface`, `cap-color`
- Características de las láminas: `gill-attachment`, `gill-spacing`, `gill-color`
- Características del tallo: `stem-surface`, `stem-color`
- Otras: `does-bruise-or-bleed`, `has-ring`, `ring-type`, `habitat`, `season`

**⚠️ IMPORTANTE:** Todas las variables categóricas usan **códigos de una sola letra**. Ver `secondary_data_meta.txt` para decodificación.

---

## 🔄 Flujo de Trabajo

### 1️⃣ Análisis Exploratorio (`AnalisisExploratorio.ipynb`)

**Objetivos:**
- Verificar formato y tipos de datos
- Identificar valores nulos y duplicados
- Detectar outliers extremos
- Identificar valores inesperados en variables categóricas

**Hallazgos Clave:**
- 4 variables con >85% de valores nulos
- 611 filas con valor 'invalid_value' en `cap-diameter`
- Códigos categóricos no documentados en `cap-surface` y `stem-root`
- Outliers biológicamente imposibles (hongos de 6 metros de diámetro)

### 2️⃣ Limpieza de Datos (`Limpieza_Datos.ipynb`)

**Estrategia de Limpieza (8 pasos):**

| Paso | Acción | Filas/Cols Afectadas | Justificación |
|------|--------|---------------------|---------------|
| 1 | Eliminar duplicados iniciales | 45 filas (0.07%) | Evitar data leakage |
| 2 | Eliminar variables >85% nulos | 4 columnas | Evitar datos sintéticos |
| 3 | Imputar 'invalid_value' | 611 filas (1.00%) | Solo 1%, preservar información |
| 4 | Eliminar códigos inesperados | ~4,200 filas (6.93%) | Datos incorrectos |
| 5 | Eliminar outliers (IQR × 3) | ~100 filas | Biológicamente imposibles |
| 6 | Imputar nulos restantes | Variable | Preservar diferencias e/p |
| 7 | Eliminar filas muy incompletas | ~5 filas | Casos irrecuperables |
| 8 | Eliminar duplicados finales | 38 filas | Después de transformaciones |

**Resultado:** Dataset 100% completo sin valores nulos, sin duplicados, sin outliers extremos.

### 3️⃣ Análisis Visual (`Analisis_Visualizaciones.ipynb`)

**Análisis Realizados:**
- Distribuciones de variables numéricas por clase
- Patrones en variables categóricas
- Correlaciones entre features
- Análisis multivariado (Pair plots, scatter plots)
- Patrones por hábitat y estación

**Insights Principales:**
- Diferencias morfológicas significativas entre hongos comestibles y venenosos
- Variables más predictivas: `stem-surface`, `cap-diameter`, `stem-width`
- Multicolinealidad detectada: `cap-diameter` ↔ `stem-width` (r=0.747)
- Hábitats peligrosos: `p` (100% venenosos), `g` (70.2% venenosos)

### 4️⃣ Modelo Predictivo (`Modelo_Predictivo.ipynb`) ✅

**Pipeline Completo:**

1. **Preprocesamiento**
   - Encoding de variables categóricas (Label Encoding)
   - Estandarización de variables numéricas
   - Split train/test (80/20) con estratificación

2. **Comparación de Modelos**
   - 6 algoritmos evaluados: LR, DT, RF, GB, SVM, KNN
   - Métrica principal: **Recall para clase venenosa**

3. **Optimización**
   - GridSearchCV con validación cruzada estratificada (5-fold)
   - Optimización de hiperparámetros del mejor modelo

4. **Evaluación**
   - Métricas estándar (accuracy, precision, recall, F1)
   - Curvas ROC y Precision-Recall
   - Feature importance
   - Análisis de threshold para minimizar falsos negativos

---

## 🏆 Resultados del Modelo

### Mejor Modelo: Random Forest (Optimizado)

| Métrica | Valor | Evaluación |
|---------|-------|------------|
| **Accuracy** | 99.27% | Excelente |
| **Precision** | 99.94% | Excelente |
| **Recall (poisonous)** | 98.83% | ⚠️ Bueno (objetivo: >99%) |
| **F1-Score** | 99.38% | Excelente |
| **AUC-ROC** | 0.9980 | Excelente |

### Hiperparámetros Optimizados

```python
{
    'n_estimators': 300,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}
```

### Análisis de Errores (Test Set: 10,709 hongos)

| Tipo de Error | Cantidad | % | Impacto |
|---------------|----------|---|---------|
| **Falsos Negativos (FN)** | 74 | 1.17% | 🔴 **CRÍTICO** - Hongos venenosos clasificados como comestibles |
| **Falsos Positivos (FP)** | 4 | 0.09% | 🟢 Aceptable - Hongos comestibles rechazados |

**Interpretación:**
- De 6,308 hongos venenosos: 6,234 detectados (98.83%), 74 no detectados
- De 4,401 hongos comestibles: 4,397 correctos (99.91%), 4 rechazados

### Top 5 Features Más Importantes

1. **stem-surface** (46.0%) - Textura del tallo
2. **cap-surface** (11.3%) - Textura del sombrero
3. **stem-width** (7.9%) - Ancho del tallo
4. **stem-height** (5.5%) - Altura del tallo
5. **cap-diameter** (4.7%) - Diámetro del sombrero

### Recomendación Final

**Nivel de Confianza:** 🟡 **AMARILLO**

**Decisión:** Modelo RECOMENDADO con precauciones adicionales

**Justificación:**
- ✅ Recall > 95% (98.83%)
- ❌ Recall < 99% (objetivo ideal)
- ⚠️ 74 falsos negativos presentes

**Uso Recomendado:**
- Implementar sistema de doble verificación para casos dudosos
- Usar threshold ajustado (0.3-0.4) para reducir FN
- Validación obligatoria con expertos micólogos antes de uso en producción
- **NO usar como única herramienta de clasificación en campo**

---

## 💻 Instalación y Ejecución

### Requisitos

```bash
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

### Instalación

```bash
# Clonar/Descargar el proyecto
cd Proy_AD

# Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# O usar requirements.txt si existe
pip install -r requirements.txt
```

### Ejecución de Notebooks

```bash
# Iniciar Jupyter Notebook
jupyter notebook

# Ejecutar en orden:
# 1. AnalisisExploratorio.ipynb
# 2. Limpieza_Datos.ipynb
# 3. Analisis_Visualizaciones.ipynb
# 4. Modelo_Predictivo.ipynb
```

### Cargar Datos en Python

```python
import pandas as pd

# Dataset original (para exploración)
df_original = pd.read_csv('MushroomDataset/MushroomDataset.csv')

# Dataset limpio (para modelado)
df_clean = pd.read_csv('MushroomDataset/MushroomDataset_cleaned.csv', sep=';')
```

**⚠️ IMPORTANTE:**
- Dataset original: usa delimitador **coma** (`,`)
- Dataset limpio: usa delimitador **punto y coma** (`;`)

---

## 🔍 Archivos Principales

### Notebooks de Análisis

| Archivo | Descripción | Estado |
|---------|-------------|--------|
| **AnalisisExploratorio.ipynb** | Exploración inicial y detección de problemas | ✅ Completo |
| **Limpieza_Datos.ipynb** | Proceso de limpieza en 8 pasos | ✅ Completo |
| **Analisis_Visualizaciones.ipynb** | Visualizaciones y patrones | ✅ Completo |
| **Modelo_Predictivo.ipynb** | Modelo ML completo con optimización | ✅ Completo |
| **Codigo.ipynb** | Notebook legacy (análisis anterior) | 📦 Archivo |

### Datos

| Archivo | Filas | Columnas | Delimiter | Descripción |
|---------|-------|----------|-----------|-------------|
| **MushroomDataset.csv** | 61,079 | 21 | `,` | Dataset original con problemas |
| **MushroomDataset_cleaned.csv** | 53,541 | 17 | `;` | Dataset limpio para ML |
| **secondary_data.csv** | 61,070 | 21 | `,` | Copia alternativa (10 filas menos) |
| **primary_data.csv** | 173 | 21 | `,` | Dataset fuente (1 fila por especie) |

### Documentación

- **CLAUDE.md**: Documentación técnica completa del proyecto
- **secondary_data_meta.txt**: Metadatos oficiales de UCI con codificación de variables
- **primary_data_meta.txt**: Metadatos del dataset primario

---

## ⚠️ Problemas de Calidad de Datos

### Problemas Identificados en Dataset Original

| Problema | Variable(s) | Magnitud | Solución |
|----------|------------|----------|----------|
| Valores nulos >85% | `veil-type`, `spore-print-color`, `veil-color`, `stem-root` | 4 variables | ❌ Eliminar variables |
| Valores 'invalid_value' | `cap-diameter` | 611 filas (1%) | ✅ Imputar con mediana por clase |
| Códigos no documentados | `cap-surface` (d), `stem-root` (f) | ~4,200 filas (7%) | ❌ Eliminar filas |
| Outliers extremos | `cap-diameter`, `stem-width` | ~100 filas | ❌ Eliminar (IQR × 3) |
| Duplicados | Todas | 45 filas (0.07%) | ❌ Eliminar |

### Validación del Dataset Limpio

✅ **Checklist Completo:**
- [x] No 'invalid_value' strings
- [x] cap-diameter es numérico (float64)
- [x] Zero valores nulos (100% completo)
- [x] No outliers >100cm o >200mm
- [x] No códigos categóricos no documentados
- [x] No duplicados
- [x] Índice limpio (0 a 53,540)
- [x] >50% de datos originales retenidos (87.66%)
- [x] Clases balanceadas (59% p, 41% e)

---

## 📊 Entregables del Proyecto

### Requeridos

- [ ] **Informe (PDF)** con:
  - Título, autores e índice
  - Descripción del problema
  - Descripción del dataset
  - Análisis exploratorio
  - Preprocesamiento de datos
  - Análisis con visualizaciones
  - Construcción del modelo predictivo
  - Resultados
  - Conclusiones

- [ ] **Presentación** (máximo 10 minutos)

- [x] **Código** (archivos .ipynb) ✅

### Archivos de Código

```python
# Para generar .py desde notebooks
jupyter nbconvert --to script AnalisisExploratorio.ipynb
jupyter nbconvert --to script Limpieza_Datos.ipynb
jupyter nbconvert --to script Modelo_Predictivo.ipynb
```

---

## 🎯 Conclusiones Principales

### Hallazgos Técnicos

1. **Calidad de Datos**: El dataset original requirió limpieza extensiva (eliminación de 12.34% de filas)

2. **Variables Críticas**: La textura del tallo (`stem-surface`) es el predictor más importante (46% de importancia)

3. **Modelo Final**: Random Forest logra 99.27% accuracy y 98.83% recall para clase venenosa

4. **Riesgo Residual**: 74 falsos negativos (1.17%) representan un riesgo de salud pública que requiere mitigación

### Limitaciones

⚠️ **IMPORTANTE - Leer antes de usar:**

1. **Datos Sintéticos**: El dataset contiene muestras hipotéticas, no recolecciones reales
2. **Generalización**: No validado con especies fuera del dataset
3. **Falsos Negativos**: El modelo todavía produce errores críticos
4. **Contexto Geográfico**: Dataset no especifica distribución geográfica
5. **Validación Experta**: Requiere validación obligatoria con micólogos profesionales

### Recomendaciones de Uso

✅ **SÍ usar para:**
- Investigación académica
- Aprendizaje de técnicas de ML
- Desarrollo de prototipos
- Análisis de patrones morfológicos

❌ **NO usar para:**
- Clasificación directa en campo sin validación
- Decisiones de consumo sin consulta experta
- Aplicaciones de producción sin mitigación de riesgos
- Educación sobre identificación de hongos salvajes

---

## 👥 Autores

Este proyecto fue desarrollado como parte del curso de Análisis de Datos.

**Equipo:**
- [Agregar nombres de los miembros del equipo]

**Institución:** Universidad de Monterrey
**Semestre:** Sexto Semestre
**Fecha:** 2024

---

## 📝 Notas Adicionales

### Decodificación de Variables Categóricas

**Colores:**
- brown=n, buff=b, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y, blue=l, orange=o, black=k

**Formas del Sombrero:**
- bell=b, conical=c, convex=x, flat=f, sunken=s, spherical=p, others=o

**Hábitat:**
- grasses=g, leaves=l, meadows=m, paths=p, heaths=h, urban=u, waste=w, woods=d

**Estaciones:**
- spring=s, summer=u, autumn=a, winter=w

**Ver `secondary_data_meta.txt` para codificación completa de todas las variables.**

### Contacto y Soporte

Para preguntas sobre el proyecto:
1. Revisar `CLAUDE.md` para documentación técnica detallada
2. Consultar los notebooks en orden de ejecución
3. Verificar que los delimitadores de CSV sean correctos
4. Asegurar que todas las validaciones pasen en `Limpieza_Datos.ipynb`

---

## 📄 Licencia

Este proyecto es desarrollado con fines académicos. El dataset proviene del UCI Machine Learning Repository.

**Dataset Citation:**
- UCI Machine Learning Repository - Secondary Mushroom Dataset
- [Agregar citación completa si está disponible]

---

**Última actualización:** Noviembre 2024

**Status del Proyecto:** ✅ **COMPLETO** - Modelo entrenado y documentado
