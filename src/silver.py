import polars as pl
import json
import re
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any
import datetime
from tqdm import tqdm

from .config import config

# Configurar logging para guardar en archivo
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "silver.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SilverTransformer:
    """Transforma datos de Bronze a Silver con limpieza y renombrado a español"""
    
    def __init__(self):
        # Cargar esquemas
        with open(config.BRONZE_SCHEMA_PATH, 'r') as f:
            self.bronze_schema = json.load(f)
        
        with open(config.SILVER_SCHEMA_PATH, 'r') as f:
            self.silver_schema = json.load(f)
        
        # Mapeo de columnas inglés -> español
        self.column_mapping = {
            "id": "id",
            "title": "titulo",
            "selftext": "texto",
            "author": "autor",
            "author_fullname": "autor_nombre_completo",
            "author_premium": "autor_premium",
            "author_patreon_flair": "autor_patreon",
            "author_flair_text": "autor_flair_texto",
            "author_flair_css_class": "autor_flair_css",
            "author_flair_type": "autor_flair_tipo",
            "subreddit": "subreddit_nombre",
            "subreddit_id": "subreddit_id",
            "subreddit_name_prefixed": "subreddit_prefijado",
            "subreddit_type": "subreddit_tipo",
            "subreddit_subscribers": "subreddit_suscriptores",
            "created_utc": "fecha_creacion_utc",
            "created_ts": "fecha_creacion",
            "edited_bool": "editado",
            "edited_utc": "editado_fecha",
            "distinguished": "distinguido",
            "is_self": "es_texto",
            "over_18": "es_adulto",
            "spoiler": "es_spoiler",
            "locked": "esta_bloqueado",
            "archived": "archivado",
            "quarantine": "en_cuarentena",
            "stickied": "esta_fijado",
            "is_original_content": "es_original",
            "is_meta": "es_meta",
            "is_crosspostable": "es_compartible",
            "score": "puntaje",
            "ups": "votos_positivos",
            "downs": "votos_negativos",
            "upvote_ratio": "ratio_upvotes",
            "num_comments": "total_comentarios",
            "num_crossposts": "total_republicaciones",
            "view_count": "vistas",
            "visited": "visitado",
            "hide_score": "ocultar_puntaje",
            "send_replies": "enviar_respuestas",
            "can_mod_post": "puede_mod",
            "suggested_sort": "orden_sugerido",
            "domain": "dominio",
            "url": "url",
            "url_overridden_by_dest": "url_destino",
            "permalink": "enlace_permanente",
            "thumbnail": "miniatura",
            "thumbnail_height": "miniatura_alto",
            "thumbnail_width": "miniatura_ancho",
            "post_hint": "pista_contenido",
            "is_video": "es_video",
            "is_gallery": "es_galeria",
            "media_only": "solo_media",
            "media": "media_json",
            "secure_media": "media_segura_json",
            "media_embed": "media_embed_json",
            "secure_media_embed": "media_segura_embed_json",
            "preview": "previsualizacion_json",
            "gallery_data": "datos_galeria_json",
            "media_metadata": "metadatos_media_json",
            "link_flair_text": "flair_texto",
            "link_flair_css_class": "flair_css",
            "link_flair_type": "flair_tipo",
            "link_flair_richtext": "flair_richtext_json",
            "author_flair_richtext": "autor_flair_richtext_json",
            "crosspost_parent": "padre_republicacion",
            "crosspost_parent_list": "lista_republicaciones_json",
            "parent_whitelist_status": "estado_lista_blanca_padre",
            "whitelist_status": "estado_lista_blanca",
            "wls": "wls",
            "pwls": "pwls",
            "content_categories": "categorias_contenido_json",
            "discussion_type": "tipo_discusion",
            "treatment_tags": "etiquetas_tratamiento_json",
            "is_robot_indexable": "indexable_por_robot",
            "author_is_blocked": "autor_bloqueado",
            "awarders": "premiadores_json",
            "all_awardings": "premiaciones_json",
            "total_awards_received": "total_premios",
            "gilded": "gilded",
            "gildings": "gildings_json",
            "poll_data": "datos_encuesta_json"
        }
    
    def clean_text(self, text: str) -> str:
        """Limpia texto básico"""
        if not text or text.strip() == "":
            return ""
        
        # Convertir a lowercase
        text = text.lower()
        
        # Remover URLs simples
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remover mentions simples
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'/u/\w+', '', text)
        text = re.sub(r'u/\w+', '', text)
        
        # Limpiar espacios múltiples
        text = re.sub(r'\\s+', ' ', text)
        
        return text.strip()
    
    def _handle_inconsistent_types(self, dfs: list[pl.DataFrame]) -> tuple[list[pl.DataFrame], dict]:
        """Maneja tipos inconsistentes en columnas y retorna DataFrames ajustados."""
        all_columns = set(col for df in dfs for col in df.columns)
        column_types = {}
        type_conflicts = {}

        # Analizar tipos inconsistentes una sola vez
        for col in all_columns:
            types = set(df[col].dtype for df in dfs if col in df.columns)
            if len(types) > 1:
                type_conflicts[col] = types
                # Priorizar tipos en orden: String > Float64 > Int64 > Boolean > Null
                if pl.Utf8 in types:
                    column_types[col] = pl.Utf8
                elif pl.Float64 in types:
                    column_types[col] = pl.Float64
                elif pl.Int64 in types:
                    column_types[col] = pl.Int64
                elif pl.Boolean in types:
                    column_types[col] = pl.Boolean
                else:
                    column_types[col] = pl.Utf8
            else:
                column_types[col] = types.pop()

        # Log consolidado de conflictos de tipos
        if type_conflicts:
            logger.warning(f"Detectados conflictos de tipos en {len(type_conflicts)} columnas:")
            for col, types in type_conflicts.items():
                logger.warning(f"  - {col}: {types} → {column_types[col]}")

        # Aplicar conversiones de tipo
        for i, df in enumerate(dfs):
            for col, target_dtype in column_types.items():
                if col in df.columns and df[col].dtype != target_dtype:
                    try:
                        if target_dtype == pl.Float64:
                            # Manejo especial para números con tipos inconsistentes
                            df = df.with_columns(
                                pl.when(pl.col(col).is_null())
                                .then(0.0)
                                .otherwise(
                                    pl.col(col)
                                    .cast(pl.Utf8)
                                    .str.replace_all(r"[^0-9.-]", "")
                                    .str.replace(r"^-", "-")
                                    .cast(pl.Float64, strict=False)
                                    .fill_null(0.0)
                                )
                                .alias(col)
                            )
                        elif target_dtype == pl.Int64:
                            df = df.with_columns(
                                pl.when(pl.col(col).is_null())
                                .then(0)
                                .otherwise(
                                    pl.col(col)
                                    .cast(pl.Utf8)
                                    .str.replace_all(r"[^0-9-]", "")
                                    .str.replace(r"^-", "-")
                                    .cast(pl.Int64, strict=False)
                                    .fill_null(0)
                                )
                                .alias(col)
                            )
                        else:
                            df = df.with_columns(
                                pl.col(col).cast(target_dtype, strict=False).fill_null("").alias(col)
                            )
                    except Exception as e:
                        logger.error(f"Error convirtiendo tipo en columna '{col}' del archivo {i}: {e}")
                        # Fallback: convertir a string
                        df = df.with_columns(pl.col(col).cast(pl.Utf8, strict=False).alias(col))
            dfs[i] = df

        return dfs, column_types

    def transform_bronze_to_silver(self) -> None:
        """Transforma todos los archivos Bronze a Silver"""
        
        logger.info("Iniciando transformación Bronze -> Silver")
        
        # Leer todos los archivos Parquet de Bronze, incluyendo subdirectorios
        bronze_files = list(config.BRONZE_DIR.rglob("*.parquet"))
        
        if not bronze_files:
            logger.warning("No se encontraron archivos Bronze para procesar")
            return
        
        # Leer y combinar todos los archivos con barra de progreso
        dfs = []
        logger.info(f"Iniciando lectura de {len(bronze_files)} archivos Bronze...")
        
        for file in tqdm(bronze_files, desc="Procesando archivos Bronze"):
            try:
                df = pl.read_parquet(file)
                dfs.append(df)
                # Solo log para archivos con errores o si es verboso
                if len(df) == 0:
                    logger.warning(f"Archivo vacío: {file.name}")
            except Exception as e:
                logger.error(f"Error leyendo {file}: {e}")
                continue
        
        if not dfs:
            logger.error("No se pudieron leer archivos Bronze")
            return

        logger.info(f"Leídos {len(dfs)} archivos exitosamente")
        
        # Manejar tipos inconsistentes
        dfs, column_types = self._handle_inconsistent_types(dfs)
        
        # Log de schema solo una vez, del primer archivo como ejemplo
        logger.info(f"Schema de ejemplo (primer archivo): {dfs[0].schema}")
        
        # Preservar la columna 'extras' antes de concatenar
        # Verificar si la columna 'extras' existe en algún archivo
        has_extras = any('extras' in df.columns for df in dfs)
        extras_data = None
        
        if has_extras:
            logger.info("Preservando columna 'extras' con datos estructurados")
            # Extraer y combinar datos de extras
            extras_dfs = []
            for df in dfs:
                if 'extras' in df.columns:
                    extras_df = df.select(['id', 'extras'])
                    extras_dfs.append(extras_df)
            
            if extras_dfs:
                extras_data = pl.concat(extras_dfs)
                logger.info(f"Preservados {len(extras_data)} registros con datos 'extras'")
        
        # Eliminar la columna 'extras' antes de concatenar para evitar conflictos de schema
        dfs = [df.drop('extras') if 'extras' in df.columns else df for df in dfs]
        
        # Combinar todos los DataFrames
        combined_df = pl.concat(dfs, how="diagonal")  # diagonal maneja columnas faltantes
        
        # Eliminar duplicados por ID
        original_count = len(combined_df)
        combined_df = combined_df.unique("id")
        dedup_count = len(combined_df)
        
        if original_count != dedup_count:
            logger.info(f"Eliminados {original_count - dedup_count} duplicados")
        
        # Re-agregar datos de extras si existen
        if extras_data is not None:
            combined_df = combined_df.join(extras_data, on="id", how="left")
            logger.info("Columna 'extras' restaurada en el dataset final")
        
        # Calcular métricas básicas
        subreddit_counts = combined_df.group_by("subreddit").agg(
            pl.col("id").count().alias("total_posts")
        )

        # Validar rango de fechas en los datos
        try:
            min_date = combined_df.select(pl.col("created_utc").min()).to_numpy()[0][0]
            max_date = combined_df.select(pl.col("created_utc").max()).to_numpy()[0][0]
            
            min_datetime = datetime.datetime.fromtimestamp(min_date, datetime.timezone.utc)
            max_datetime = datetime.datetime.fromtimestamp(max_date, datetime.timezone.utc)
        except Exception as e:
            logger.warning(f"Error calculando rango de fechas: {e}")
            min_datetime = max_datetime = "N/A"

        # Limpiar y validar datos de manera más robusta
        logger.info("Aplicando limpieza y validación de datos...")
        
        # Normalizar columnas con valores null/inconsistentes
        for col in combined_df.columns:
            col_dtype = combined_df[col].dtype
            
            if col_dtype == pl.Null:
                combined_df = combined_df.with_columns(pl.col(col).cast(pl.Utf8).fill_null(""))
            elif col_dtype in [pl.Float64, pl.Int64] and combined_df[col].null_count() > 0:
                # Para columnas numéricas, rellenar nulls con 0
                fill_value = 0.0 if col_dtype == pl.Float64 else 0
                combined_df = combined_df.with_columns(pl.col(col).fill_null(fill_value))
            elif col_dtype == pl.Utf8 and combined_df[col].null_count() > 0:
                # Para columnas de texto, rellenar nulls con string vacío
                combined_df = combined_df.with_columns(pl.col(col).fill_null(""))
        
        logger.info(f"Dataset limpio: {len(combined_df)} registros, {len(combined_df.columns)} columnas")
        
        # Aplicar transformaciones
        transformed_df = self._apply_transformations(combined_df)
        
        # Guardar resultado
        self._save_silver_data(transformed_df)
        
        # Consolidar resumen final
        logger.info("\n=== RESUMEN DE TRANSFORMACIÓN BRONZE → SILVER ===")
        logger.info(f"📁 Archivos procesados: {len(bronze_files)}")
        logger.info(f"📊 Total registros combinados: {len(combined_df)}")
        logger.info(f"🔄 Registros únicos después de deduplicación: {dedup_count}")
        logger.info(f"📅 Rango de fechas: {min_datetime} → {max_datetime}")
        logger.info(f"📈 Distribución por subreddit:")
        logger.info(subreddit_counts)
        
        # Comparar Bronze vs Silver
        bronze_summary = combined_df.group_by("subreddit").agg(
            pl.col("id").count().alias("total_registros_bronze")
        )

        cleaned_summary = transformed_df.group_by("subreddit_nombre").agg(
            pl.col("id").count().alias("total_registros_limpios")
        )

        logger.info("\n📋 Resumen por subreddit (Bronze):")
        logger.info(bronze_summary)

        logger.info("\n✨ Resumen por subreddit (Silver):")
        logger.info(cleaned_summary)

        # Calcular y mostrar distribución con porcentajes
        total_posts = len(transformed_df)
        distribution_with_percentages = cleaned_summary.with_columns(
            (pl.col("total_registros_limpios") * 100.0 / total_posts).round(1).alias("porcentaje")
        ).sort("total_registros_limpios", descending=True)

        logger.info(f"\n🎯 Distribución Final por Subreddit ({total_posts} posts totales):")
        for row in distribution_with_percentages.iter_rows(named=True):
            subreddit = row['subreddit_nombre']
            posts = row['total_registros_limpios']
            percent = row['porcentaje']
            logger.info(f"   {subreddit}: {posts} posts ({percent}%)")

        logger.info("✅ Transformación Bronze → Silver completada exitosamente")
    
    def _cast_null_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convierte columnas con valores Null a tipos predeterminados"""
        for column in df.columns:
            if df[column].dtype == pl.Null:
                df = df.with_columns(pl.col(column).cast(pl.Utf8))
        return df

    def _apply_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aplica todas las transformaciones de limpieza y renombrado"""
        
        # Convertir columnas con valores Null a tipos predeterminados
        df = self._cast_null_columns(df)
        
        # Aplicar limpieza de texto
        df = df.with_columns([
            pl.col("title").map_elements(self.clean_text, return_dtype=pl.String).alias("titulo_limpio"),
            pl.col("selftext").map_elements(self.clean_text, return_dtype=pl.String).alias("texto_limpio")
        ])
        
        # Convertir timestamps a datetime
        df = df.with_columns([
            pl.from_epoch(pl.col("created_utc"), time_unit="s").alias("fecha_creacion_dt")
        ])
        
        # Renombrar columnas según mapeo
        rename_dict = {}
        for old_col, new_col in self.column_mapping.items():
            if old_col in df.columns:
                rename_dict[old_col] = new_col
        
        df = df.rename(rename_dict)
        
        # Agregar columnas de fecha_creacion en formato datetime
        if "fecha_creacion_dt" in df.columns:
            df = df.drop("fecha_creacion_dt")
        
        # Usar created_ts que ya viene en formato ISO
        if "created_ts" in df.columns:
            df = df.with_columns([
                pl.col("created_ts").str.to_datetime().alias("fecha_creacion")
            ])
        
        # Eliminar columnas complejas solo si es necesario para el análisis final
        # Nota: 'extras' se preservó en el dataset principal como datos estructurados
        columns_to_drop = []
        if "extras" in df.columns:
            # Solo eliminar extras si causa problemas en la escritura final
            logger.info("Columna 'extras' presente en transformación final - preservando datos estructurados")
        
        # Manejar columnas con valores Null reemplazándolos por 'NA'
        columns_with_nulls = [
            "edited_utc", "distinguished", "view_count", "gallery_data",
            "crosspost_parent", "parent_whitelist_status", "whitelist_status",
            "poll_data", "discussion_type"
        ]

        for col in columns_with_nulls:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null("NA").cast(pl.Utf8))
        
        return df
    
    def _save_silver_data(self, df: pl.DataFrame) -> None:
        """Guarda datos transformados en capa Silver"""
        
        # Asegurar que el directorio existe
        config.SILVER_DIR.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo (overwrite)
        filepath = config.SILVER_DIR / "silver.parquet"
        df.write_parquet(filepath)
        
        logger.info(f"Guardados {len(df)} registros en {filepath}")

def main():
    """Función principal para ejecutar transformación Silver"""
    
    transformer = SilverTransformer()
    transformer.transform_bronze_to_silver()

if __name__ == "__main__":
    main()
