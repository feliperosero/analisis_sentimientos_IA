import polars as pl
import pandas as pd
import re
from pathlib import Path
import logging
from typing import Dict, Set, List

from .config import config

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldFeaturesGenerator:
    """Genera features A (baseline) y B (ontológicas) desde datos Silver"""
    
    def __init__(self):
        # Cargar diccionarios ontológicos
        self.conceptos_es = self._load_concepts("conceptos_es.csv")
        self.conceptos_en = self._load_concepts("conceptos_en.csv")
        
    def _load_concepts(self, filename: str) -> Dict[str, Set[str]]:
        """Carga diccionarios de conceptos desde CSV"""
        
        filepath = Path(__file__).parent / "ontology" / "diccionarios" / filename
        
        if not filepath.exists():
            logger.warning(f"Archivo de conceptos no encontrado: {filepath}")
            return {}
        
        df = pd.read_csv(filepath)
        conceptos = {}
        
        for _, row in df.iterrows():
            concepto = row['concepto']
            sinonimo = row['sinonimo'].lower()
            
            if concepto not in conceptos:
                conceptos[concepto] = set()
            
            conceptos[concepto].add(sinonimo)
        
        logger.info(f"Cargados {len(conceptos)} conceptos desde {filename}")
        return conceptos
    
    def generate_features_a(self, df: pl.DataFrame) -> pl.DataFrame:
        """Genera features baseline (A)"""
        
        logger.info("Generando features A (baseline)")
        
        # Primero convertir fecha_creacion a datetime si es string
        if df['fecha_creacion'].dtype == pl.String:
            df = df.with_columns([
                pl.col("fecha_creacion").str.to_datetime().alias("fecha_creacion")
            ])
        
        features_a = df.with_columns([
            # Características de longitud
            pl.col("titulo").str.len_chars().alias("long_titulo"),
            pl.col("texto").str.len_chars().alias("long_texto"),
            pl.col("titulo").str.split(" ").list.len().alias("palabras_titulo"),
            pl.col("texto").str.split(" ").list.len().alias("palabras_texto"),
            
            # Características temporales
            pl.col("fecha_creacion").dt.hour().alias("hora_dia"),
            pl.col("fecha_creacion").dt.weekday().alias("dia_semana"),
            
            # Características de engagement
            (pl.col("puntaje") + pl.col("total_comentarios")).alias("interaccion_proxy"),
            (pl.col("puntaje") / pl.col("total_comentarios").clip(1)).alias("ratio_puntos_comentarios"),
            
            # Características textuales simples
            pl.col("titulo").str.contains(r"\\?").alias("contiene_pregunta"),
            pl.col("titulo").str.to_uppercase().eq(pl.col("titulo")).alias("titulo_mayusculas"),
            pl.col("es_adulto").cast(pl.Int8).alias("es_adulto_int"),
            pl.col("es_video").cast(pl.Int8).alias("es_video_int"),
            
            # Características del subreddit
            pl.col("subreddit_suscriptores").fill_null(0).alias("suscriptores_subreddit")
        ])
        
        return features_a
    
    def generate_features_b(self, df: pl.DataFrame) -> pl.DataFrame:
        """Genera features ontológicas (B) = A + características ontológicas"""
        
        logger.info("Generando features B (ontológicas)")
        
        # Primero generar features A
        features_b = self.generate_features_a(df)
        
        # Agregar características ontológicas
        ontological_features = []
        
        # Para cada concepto, contar ocurrencias en título y texto
        all_concepts = set(self.conceptos_es.keys()) | set(self.conceptos_en.keys())
        
        for concepto in all_concepts:
            # Obtener sinónimos en ambos idiomas
            sinonimos = set()
            if concepto in self.conceptos_es:
                sinonimos.update(self.conceptos_es[concepto])
            if concepto in self.conceptos_en:
                sinonimos.update(self.conceptos_en[concepto])
            
            if not sinonimos:
                continue
            
            # Crear patrón regex para buscar palabras completas
            pattern = r'\\b(' + '|'.join(re.escape(s) for s in sinonimos) + r')\\b'
            
            # Contar en título
            features_b = features_b.with_columns([
                pl.col("titulo").str.count_matches(pattern).alias(f"n_{concepto}_titulo")
            ])
            
            # Contar en texto
            features_b = features_b.with_columns([
                pl.col("texto").str.count_matches(pattern).alias(f"n_{concepto}_texto")
            ])
            
            # Flag binario si contiene el concepto
            features_b = features_b.with_columns([
                (pl.col(f"n_{concepto}_titulo") + pl.col(f"n_{concepto}_texto") > 0).alias(f"es_{concepto}")
            ])
        
        # Características agregadas ontológicas
        concept_titulo_cols = [col for col in features_b.columns if col.startswith("n_") and col.endswith("_titulo")]
        concept_texto_cols = [col for col in features_b.columns if col.startswith("n_") and col.endswith("_texto")]
        
        if concept_titulo_cols:
            features_b = features_b.with_columns([
                pl.sum_horizontal(concept_titulo_cols).alias("total_conceptos_titulo"),
                pl.sum_horizontal(concept_texto_cols).alias("total_conceptos_texto")
            ])
            
            # Diversidad de conceptos
            features_b = features_b.with_columns([
                (pl.sum_horizontal([pl.col(col) > 0 for col in concept_titulo_cols])).alias("diversidad_conceptos_titulo"),
                (pl.sum_horizontal([pl.col(col) > 0 for col in concept_texto_cols])).alias("diversidad_conceptos_texto")
            ])
        
        return features_b
    
    def process_silver_to_gold(self) -> None:
        """Procesa datos Silver y genera features A y B"""
        
        # Leer datos Silver
        silver_path = config.SILVER_DIR / "silver.parquet"
        
        if not silver_path.exists():
            logger.error(f"Archivo Silver no encontrado: {silver_path}")
            return
        
        df = pl.read_parquet(silver_path)
        logger.info(f"Leídos {len(df)} registros desde Silver")
        
        # Generar features A
        features_a = self.generate_features_a(df)
        
        # Generar features B
        features_b = self.generate_features_b(df)
        
        # Guardar features
        self._save_features(features_a, "features_A.parquet")
        self._save_features(features_b, "features_B.parquet")
        
        logger.info("Generación de features completada")
    
    def _save_features(self, df: pl.DataFrame, filename: str) -> None:
        """Guarda features en capa Gold"""
        
        # Asegurar que el directorio existe
        config.GOLD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo
        filepath = config.GOLD_DIR / filename
        df.write_parquet(filepath)
        
        logger.info(f"Guardadas {len(df)} filas de features en {filepath}")

def main():
    """Función principal para generar features"""
    
    generator = GoldFeaturesGenerator()
    generator.process_silver_to_gold()

if __name__ == "__main__":
    main()
