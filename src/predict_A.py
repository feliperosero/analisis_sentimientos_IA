import polars as pl
import joblib
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import config

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictorA:
    """Predictor usando features A (baseline)"""
    
    def __init__(self):
        self.model_path = config.MODELS_DIR / "modelo_a.pkl"
        self.model = None
        
    def load_or_create_model(self):
        """Carga modelo existente o crea uno placeholder"""
        
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Modelo A cargado desde {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Error cargando modelo A: {e}")
        
        # Crear modelo placeholder simple
        logger.info("Creando modelo A placeholder")
        self.model = LogisticRegression(random_state=42, class_weight='balanced')
        
        # Datos sintéticos para inicializar
        X_dummy = np.random.rand(100, 10)  # 10 features principales
        y_dummy = np.random.choice([0, 1], 100)  # Clasificación binaria
        
        self.model.fit(X_dummy, y_dummy)
        
        # Guardar modelo placeholder
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Modelo A placeholder guardado en {self.model_path}")
    
    def predict(self) -> None:
        """Ejecuta predicciones con features A"""
        
        logger.info("Iniciando predicciones A (baseline)")
        
        # Cargar features A
        features_path = config.GOLD_DIR / "features_A.parquet"
        
        if not features_path.exists():
            logger.error(f"Features A no encontradas: {features_path}")
            return
        
        df = pl.read_parquet(features_path)
        logger.info(f"Cargadas {len(df)} filas de features A")
        
        # Cargar o crear modelo
        self.load_or_create_model()
        
        # Preparar features para predicción
        feature_columns = [
            "long_titulo", "long_texto", "palabras_titulo", "palabras_texto",
            "hora_dia", "dia_semana", "interaccion_proxy", "ratio_puntos_comentarios",
            "es_adulto_int", "es_video_int"
        ]
        
        # Filtrar columnas que existen
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if len(available_columns) < 5:
            logger.error(f"Insuficientes features disponibles: {available_columns}")
            return
        
        # Preparar matriz de features
        X = df.select(available_columns).fill_null(0).to_numpy()
        
        # Si el modelo espera más features, rellenar con ceros
        if X.shape[1] < 10:
            X_padded = np.zeros((X.shape[0], 10))
            X_padded[:, :X.shape[1]] = X
            X = X_padded
        
        # Realizar predicciones
        try:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Crear DataFrame de resultados
            results_df = df.select(["id", "fecha_creacion", "subreddit_nombre", "titulo"]).with_columns([
                pl.Series("prediccion", predictions.astype(int)),
                pl.Series("probabilidad", probabilities[:, 1]),
                pl.lit("modelo_a").alias("modelo"),
                pl.lit(datetime.now().isoformat()).alias("ts_prediccion")
            ])
            
            # Guardar resultados
            self._save_predictions(results_df)
            
            logger.info(f"Predicciones A completadas: {len(results_df)} registros")
            
        except Exception as e:
            logger.error(f"Error en predicciones A: {e}")
    
    def _save_predictions(self, df: pl.DataFrame) -> None:
        """Guarda predicciones en capa Reports"""
        
        # Asegurar que el directorio existe
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo
        filepath = config.REPORTS_DIR / "predicciones_a.parquet"
        df.write_parquet(filepath)
        
        logger.info(f"Predicciones A guardadas en {filepath}")

def main():
    """Función principal para ejecutar predicciones A"""
    
    predictor = PredictorA()
    predictor.predict()

if __name__ == "__main__":
    main()
