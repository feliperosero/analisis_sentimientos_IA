# Análisis de Sentimientos con IA

Un proyecto de clasificación de sentimientos usando modelos de transformers y PyTorch Lightning.

## Descripción

Este proyecto implementa un sistema de análisis de sentimientos utilizando el modelo XLM-RoBERTa para clasificar textos en español en las siguientes categorías:
- Positivo
- Negativo  
- Neutral

## Estructura del Proyecto

```
├── data/
│   └── silver/          # Datos procesados para entrenamiento
├── src/                 # Código fuente del proyecto
│   ├── config.py        # Configuración del proyecto
│   ├── ingest.py        # Ingesta de datos
│   ├── silver.py        # Procesamiento de datos
│   └── features/        # Extracción de características
├── notebooks/           # Notebooks de experimentación
│   └── 04_silver_sentimiento_dl.ipynb  # Notebook principal
├── artifacts/           # Modelos y métricas guardados
├── outputs/             # Resultados y datos procesados
└── README.md
```

## Características

- Procesamiento de datos multiidioma con detección automática
- Balanceo de clases con pesos calculados automáticamente
- Entrenamiento con PyTorch Lightning para escalabilidad
- Validación cruzada y métricas detalladas
- Pipeline de inferencia optimizado
- Guardado automático de checkpoints del mejor modelo

## Requisitos

- Python 3.8+
- PyTorch
- PyTorch Lightning
- Transformers (Hugging Face)
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/feliperosero/analisis_sentimientos_IA.git
cd analisis_sentimientos_IA
```

2. Crea un entorno virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # En Linux/Mac
# o
.venv\Scripts\activate     # En Windows
```

3. Instala las dependencias:
```bash
pip install torch pytorch-lightning transformers pandas scikit-learn matplotlib seaborn tqdm langdetect pyarrow
```

## Uso

### Entrenamiento

Ejecuta el notebook principal `notebooks/04_silver_sentimiento_dl.ipynb` para:
1. Cargar y preprocesar los datos
2. Configurar el modelo XLM-RoBERTa
3. Entrenar con validación cruzada
4. Evaluar métricas de rendimiento
5. Guardar el modelo entrenado

### Inferencia

Utiliza la clase `SentimentInferencePipeline` para hacer predicciones:

```python
from src.predict_A import SentimentInferencePipeline

pipeline = SentimentInferencePipeline.load_from_checkpoint("ruta/al/modelo.ckpt")
result = pipeline.predict("Este es un texto de ejemplo")
print(f"Sentimiento: {result['label']}, Confianza: {result['confidence']:.3f}")
```

## Métricas de Rendimiento

El modelo ha sido evaluado utilizando:
- Precisión (Precision)
- Recall
- F1-Score
- Matriz de confusión
- Distribución de confianza por clase

Los resultados se guardan automáticamente en `artifacts/` y se generan visualizaciones en `artifacts/plots/`.

## Datos

Los datos se procesan en tres niveles:
- **Bronze**: Datos crudos originales
- **Silver**: Datos limpiados y preprocesados
- **Gold**: Datos finales listos para modelado

## Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

## Autor

Felipe Rosero
