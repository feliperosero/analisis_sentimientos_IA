import polars as pl
import joblib
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config import config

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictorB:
    """Predictor usando features B (ontológicas)"""
    
    def __init__(self):
        self.model_path = config.MODELS_DIR / "modelo_b.pkl"
        self.model = None
        
    def load_or_create_model(self, n_features=None):
        """Carga modelo existente o crea uno placeholder"""
        
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Modelo B cargado desde {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Error cargando modelo B: {e}")
        
        # Crear modelo placeholder más complejo
        logger.info("Creando modelo B placeholder")
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # Usar n_features dinámico si se proporciona
        if n_features is None:
            n_features = 25
        
        # Datos sintéticos para inicializar
        X_dummy = np.random.rand(100, n_features)
        y_dummy = np.random.choice([0, 1], 100)  # Clasificación binaria
        
        self.model.fit(X_dummy, y_dummy)
        
        # Guardar modelo placeholder
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Modelo B placeholder guardado en {self.model_path}")
    
    def predict(self) -> None:
        """Ejecuta predicciones con features B"""
        
        logger.info("Iniciando predicciones B (ontológicas)")
        
        # Cargar features B
        features_path = config.GOLD_DIR / "features_B.parquet"
        
        if not features_path.exists():
            logger.error(f"Features B no encontradas: {features_path}")
            return
        
        df = pl.read_parquet(features_path)
        logger.info(f"Cargadas {len(df)} filas de features B")
        
        # Preparar features para predicción (baseline + ontológicas)
        feature_columns = [col for col in df.columns 
                          if col.startswith(("long_", "palabras_", "hora_", "dia_", "interaccion_", 
                                           "ratio_", "es_", "n_", "total_", "diversidad_"))]
        
        # Remover duplicados manteniendo orden
        feature_columns = list(dict.fromkeys(feature_columns))
        
        logger.info(f"Features seleccionadas para modelo B: {len(feature_columns)}")
        
        if len(feature_columns) < 5:
            logger.error(f"Insuficientes features disponibles: {feature_columns}")
            return
        
        # Preparar matriz de features
        X = df.select(feature_columns).fill_null(0).to_numpy()
        
        # Cargar o crear modelo con el número correcto de features
        self.load_or_create_model(n_features=X.shape[1])
        
        # Si el modelo no ha sido entrenado, crear datos dummy para fit
        if not hasattr(self.model, 'classes_'):
            logger.info(f"Entrenando modelo placeholder con {X.shape[1]} features")
            # Crear datos dummy para entrenar el modelo
            y_dummy = np.random.randint(0, 2, X.shape[0])
            self.model.fit(X, y_dummy)
        
        # Realizar predicciones
        try:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Crear DataFrame de resultados
            results_df = df.select(["id", "fecha_creacion", "subreddit_nombre", "titulo"]).with_columns([
                pl.Series("prediccion", predictions.astype(int)),
                pl.Series("probabilidad", probabilities[:, 1]),
                pl.lit("modelo_b").alias("modelo"),
                pl.lit(datetime.now().isoformat()).alias("ts_prediccion")
            ])
            
            # Guardar resultados
            self._save_predictions(results_df)
            
            logger.info(f"Predicciones B completadas: {len(results_df)} registros")
            
        except Exception as e:
            logger.error(f"Error en predicciones B: {e}")
    
    def _save_predictions(self, df: pl.DataFrame) -> None:
        """Guarda predicciones en capa Reports"""
        
        # Asegurar que el directorio existe
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo
        filepath = config.REPORTS_DIR / "predicciones_b.parquet"
        df.write_parquet(filepath)
        
        logger.info(f"Predicciones B guardadas en {filepath}")

def main():
    """Función principal para ejecutar predicciones B"""
    
    predictor = PredictorB()
    predictor.predict()

if __name__ == "__main__":
    main()
