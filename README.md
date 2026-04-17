# Prestige analysis rebuild

Reconstrucción reproducible del análisis de prestigio e influencia grupal en equipos de estudiantes.

## Qué incluye

- `prestige_common.py`: utilidades compartidas, carga del Excel, auditoría, ingeniería de variables y persistencia de modelos.
- `prestige_pipeline.py`: pipeline principal con validación por grupos, optimización bayesiana de fórmulas y modelos, y exportación del Excel maestro.
- `prestige_plots.py`: generación de gráficos listos para tesis/GitHub a partir del Excel maestro.
- `prestige_shap.py`: explicabilidad SHAP de los modelos finales, con tablas y figuras.
- `CHANGELOG.md`: cambios metodológicos y técnicos respecto al código antiguo.
- `PRISM_PROMPT.md`: prompt para actualizar el informe LaTeX anterior con los nuevos resultados.

## Input esperado

El input base sigue siendo exactamente:

`consolidado_ordenado.xlsx`

No se modifica el archivo original. El pipeline conserva:

- una copia cruda del input
- una versión tipada/limpia
- una versión derivada con variables nuevas y columnas originales intactas

## Dependencias

Python 3.10+ recomendado.

Instalar:

```bash
pip install pandas numpy scipy scikit-learn networkx openpyxl matplotlib shap optuna joblib
```

## Ejecución

### 1) Pipeline principal

```bash
python prestige_pipeline.py \
  --input /ruta/a/consolidado_ordenado.xlsx \
  --output-dir prestige_outputs \
  --model-trials 30 \
  --formula-trials 80
```

### 2) Gráficos

```bash
python prestige_plots.py \
  --master-excel prestige_outputs/reports/prestige_master.xlsx \
  --plots-dir prestige_outputs/plots
```

### 3) SHAP

```bash
python prestige_shap.py \
  --models-dir prestige_outputs/models \
  --shap-dir prestige_outputs/shap
```

## Qué genera

### Excel maestro

`prestige_outputs/reports/prestige_master.xlsx`

Incluye, entre otras, hojas de:

- input crudo
- input tipado
- dataset derivado
- auditoría de columnas y rangos
- correlaciones absolutas y relativas
- mejores pesos de fórmula por target
- trials de Optuna
- métricas outer-CV por grupos
- predicciones out-of-fold
- importancias finales de variables
- tablas listas para gráficos

### Carpeta de modelos

`prestige_outputs/models/`

Un `.joblib` por target con:

- pipeline final entrenado
- parámetros óptimos
- nombres de features crudas y transformadas
- mapping transformed → original

### Carpeta SHAP

`prestige_outputs/shap/`

- `shap_outputs.xlsx`
- `summary_*.png`
- `bar_*.png`
- `dependence_*.png`
- `waterfall_*.png`

## Decisiones metodológicas clave

- Se usa `GroupKFold` para respetar la estructura por equipos.
- `Gender` y `Rol` se tratan como categóricas y se codifican dentro del `Pipeline`.
- Las columnas de prestigio se auditan antes de su uso; si ya están correctamente en `[0,1]`, se conservan, y si no, se reescalan de forma explícita.
- La optimización bayesiana se usa para:
  - pesos de la fórmula de prestigio
  - hiperparámetros del modelo predictivo
- La explicabilidad usa SHAP sobre los modelos finales.

## Recomendación práctica

Para una corrida rápida de prueba, baja los trials. Para resultados de tesis, usa al menos los valores por defecto o incluso más, dependiendo del tiempo disponible.
