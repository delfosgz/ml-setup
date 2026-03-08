# ML Project Architecture — Source of Truth

## Estructura del Proyecto

```
project/
├── .env                     # Credenciales (API keys). NUNCA a Git.
├── .gitignore               # Incluye: mlruns/, .env, __pycache__/
├── pyproject.toml + uv.lock # Gestión de entorno determinista con uv.
├── configs/                 # Parámetros del experimento. Solo YAML.
├── data/
│   ├── raw/                 # Datos originales. Inmutables.
│   ├── interim/             # Datos en transformación. Borrable y regenerable.
│   └── processed/           # Datos finales listos para el modelo.
├── notebooks/               # Solo EDA y exploración visual.
├── reports/
│   └── figures/             # Reportes HTML y gráficas de evaluación.
├── mlruns/                  # Autogenerado por MLflow. NUNCA editar a mano.
└── src/
    ├── data/                # Scripts de ingesta y partición.
    ├── features/            # Pipeline de transformación de features.
    ├── models/              # Entrenamiento, tracking y evaluación.
    └── visualization/       # Funciones reutilizables para generar figuras.
```

---

## Contratos por Directorio

### `data/raw/`
- Almacena los archivos originales tal como vienen de la fuente. Inmutables.
- Los scripts de **`src/data/`** son los únicos que escriben aquí (ingesta) y leen de aquí (partición).
- ❌ Nunca editar archivos a mano. Ningún otro directorio de `src/` escribe aquí. Nunca desde notebooks.

### `data/interim/`
- Estado transitorio entre raw y processed. Borrable y regenerable en cualquier momento.
- Exclusivo para **cambios estructurales**: split train/val/test, unión de tablas, filtrado de filas, estandarización de nombres de columnas. El valor de los datos no cambia aquí.
- Los scripts de **`src/data/`** escriben aquí. Los scripts de **`src/features/`** leen de aquí.
- ❌ Ningún notebook escribe aquí. `src/models/` nunca lee de aquí. Si el dataset ya viene listo y no requiere transformaciones estructurales, este directorio puede omitirse.

### `data/processed/`
- Datos finales listos para el modelo. Única fuente válida para entrenamiento.
- Exclusivo para **cambios de valores**: imputación, encoding, scaling, creación de features. Aquí es donde actúa el pipeline de sklearn.
- Los scripts de **`src/features/`** escriben aquí. Los scripts de **`src/models/`** leen de aquí.
- ❌ Solo `src/features/` escribe aquí. Nunca notebooks. Nunca scripts de modelos.

### `configs/`
- Contiene archivos YAML con todos los parámetros configurables del experimento: hiperparámetros, rutas relativas, semilla aleatoria, nombre del experimento MLflow, columnas de features y target.
- Leído por `src/features/` y `src/models/`.
- ❌ Solo YAML. Sin lógica de programación. Sin credenciales. Sin rutas absolutas.

### `notebooks/`
- Exclusivo para EDA y exploración visual. Nada de lo que produce alimenta el pipeline.
- Lee de `data/raw/` o `data/interim/` (nunca escribe en `data/`).
- Escribe en `reports/` (reportes HTML, gráficas exploratorias).
- ❌ Nunca entrena modelos finales. Nunca genera archivos en `data/`. Nunca se importa desde `src/`.

### `src/data/`
- Contiene scripts de **ingesta** (conexión a fuente externa, descarga, lectura de BD) y de **partición** (split train/val/test).
- Los scripts de ingesta leen de `.env` y escriben en `data/raw/`.
- Los scripts de partición leen de `data/raw/` y escriben en `data/interim/`.
- La sustitución de strings nulos (`"None"`, `"null"`, `""`, etc.) por `np.nan` ocurre aquí **únicamente** para poder calcular correctamente los ratios de nulos antes de la decisión de drop de columnas. Esta misma lógica también vive como el primer transformer del pipeline en `src/features/` para manejar strings nulos en inferencia.
- ❌ No aplica transformaciones de valores (imputación, encoding, scaling). Eso es `src/features/`. No escribe en `data/processed/`. No crea columnas derivadas.

### `src/features/`
- Contiene el pipeline de transformación: imputación, encoding, scaling, creación de nuevas features.
- Lee de `data/interim/` y `configs/`. Escribe en `data/processed/`.
- El pipeline sklearn se ajusta (`fit`) **solo sobre datos de train**. En val y test solo `transform`.
- **Regla de oro del pipeline:** antes de añadir cualquier transformación, pregunta: ¿esto puede aplicarse a una sola fila nueva en producción? Si la respuesta es no, no entra al pipeline.
  - ✅ `SimpleImputer` → sí entra. En producción sabe reemplazar nulos con la mediana aprendida del train.
  - ❌ `dropna` → no entra. En producción no puedes rechazar un input por tener nulos. El `dropna` se aplica solo sobre los datos de train en `src/data/`, antes de llegar aquí.
- **Regla de creación de features:** toda feature derivada de columnas existentes (descomposición de strings, ratios, interacciones) debe crearse como un transformer dentro del pipeline de sklearn, nunca en `src/data/` ni con pandas antes del `fit`. La única excepción es `src/data/` para operaciones estrictamente estructurales (splits, filtrado de filas). Criterio de clasificación:
  - ✅ **Entra al pipeline** — puede derivarse de una sola fila sin conocimiento externo. Ejemplos: descomponer `Cabin="B/12/P"` en `cabin_deck`, `cabin_num`, `cabin_side`; calcular `total_spend` como suma de columnas de la misma fila; extraer `group_id` de `PassengerId`.
  - ❌ **No entra al pipeline** — requiere conocimiento de múltiples filas para calcularse. Ejemplo: `group_size` requiere contar cuántos pasajeros comparten el mismo `group_id` en todo el dataset. En inferencia con una sola fila eso es imposible. Este tipo de features se descartan o se reemplazan por un valor por defecto fijo.
- **Selección de features — Filter Methods** (correlación, mutual information, chi2): la exploración vive en `notebooks/`. La decisión final (qué features conservar) se documenta en `configs/` como `selected_features`. `src/features/` simplemente lee esa lista del YAML. No requiere ningún objeto en el pipeline porque la decisión ya está tomada.
- **Selección de features — Wrapper Methods** (RFE, RFECV): aprenden del train, por lo tanto viven dentro del pipeline de sklearn en `src/features/`. El selector es un transformer más: hace `fit` sobre train y aplica la misma selección automáticamente en inferencia. Sus hiperparámetros (ej. `n_features_to_select`) van en `configs/`.
- ❌ Sin lógica de entrenamiento de modelos. Sin visualizaciones. Sin lectura desde notebooks.

### `src/models/`
- Contiene scripts de **entrenamiento** (instancia el modelo, loguea en MLflow) y de **evaluación** (carga el modelo, genera métricas y submission).
- Lee de `data/processed/` y `configs/`. Loguea en `mlruns/`. Llama funciones de `src/visualization/` para generar y loguear figuras como artifacts en MLflow.
- ❌ No hace split de datos. No lee de `data/raw/` ni `data/interim/`. No hardcodea hiperparámetros. No contiene código de visualización directamente.

### `src/visualization/`
- Contiene funciones reutilizables para generar figuras: confusion matrix, curva ROC, importancia de features, distribuciones, etc.
- Llamadas desde `src/models/`. Nunca se ejecuta de forma independiente.
- Cada función recibe datos o un modelo como input y devuelve un objeto figura.
  - Figuras logueadas como artifacts en `mlruns/` vía `mlflow.log_figure()` → ligadas a un run específico, comparables entre experimentos en la MLflow UI.
  - Figuras exportadas a `reports/figures/` → plots finales del mejor modelo para presentar a negocio.
- ❌ No lee de `data/`. No escribe directamente en ningún directorio, solo devuelve figuras. Sin lógica de entrenamiento ni transformación.

### `reports/` y `reports/figures/`
- Destino de reportes HTML (SweetViz, ydata-profiling) y gráficas finales (PNG, SVG).
- Escrito por `notebooks/` (EDA) y `src/models/` (evaluación).
- ❌ No es parte del pipeline. Ningún script de `src/` lee de aquí.

### `mlruns/`
- Gestionado 100% por MLflow. Almacena parámetros, métricas y modelos de cada run.
- ❌ Nunca editar a mano. Nunca versionar en Git. Solo interactuar vía `mlflow` API o `mlflow ui`.

---

## Flujo de Datos (Unidireccional)

```
Fuente externa
    ↓  src/data/  (ingesta)
data/raw/
    ↓  src/data/  (partición)
data/interim/
    ↓  src/features/
data/processed/
    ↓  src/models/  →  src/visualization/  →  mlruns/ (artifacts por run)
                                           →  reports/figures/ (plots finales)
```

---

## Flujo de Trabajo (Paso a Paso)

| # | Qué hacer | Quién |
|---|-----------|-------|
| 0 | Setup: `uv sync` + `uv pip install -e .` + completar `.env` | Manual |
| 1 | Descargar / ingestar datos → `data/raw/` | `src/data/` |
| 2 | Split train/val/test → `data/interim/` | `src/data/` |
| 3 | EDA visual → exportar reporte a `reports/` | `notebooks/` |
| 4 | Documentar decisiones de features en `configs/` | Manual |
| 5 | Pipeline de transformación → `data/processed/` | `src/features/` |
| 6 | Entrenar modelo + loguear en MLflow | `src/models/` |
| 7 | Evaluar + generar outputs finales en `reports/` | `src/models/` |

---

## Reglas Globales

- `data/` nunca contiene código (`.py`, `.ipynb`).
- `configs/` nunca contiene lógica de programación ni credenciales.
- `notebooks/` nunca escribe en `data/` ni genera artefactos del pipeline.
- Hiperparámetros **siempre** en `configs/`, nunca hardcodeados en `src/`.
- Credenciales **siempre** en `.env`, nunca en el código ni en `configs/`.
- El pipeline sklearn se ajusta (`fit`) **solo sobre datos de entrenamiento**.
- `mlruns/` y `.env` siempre en `.gitignore`.
- **Toda transformación de features debe vivir dentro del objeto pipeline de sklearn.** Nunca en pandas antes del `fit`, nunca en el notebook. Si la transformación no está encapsulada en el pipeline, no existirá en producción. Lo que se loguea en MLflow es `Pipeline([preprocessor, model])` completo, no el modelo solo. En producción se carga ese objeto único y recibe los datos crudos tal como los manda la fuente.