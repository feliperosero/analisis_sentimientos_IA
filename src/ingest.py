import sys
from pathlib import Path
import os

# Ensure the project directory is in the Python path
project_dir = Path(__file__).resolve().parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

import praw
import polars as pl
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from pathlib import Path
import logging
from tqdm import tqdm
import time
from dotenv import load_dotenv
import importlib
import src.config as config_module  # Importar el módulo explícitamente
import requests
from concurrent.futures import ThreadPoolExecutor
import random
from requests.exceptions import HTTPError
from colorama import Fore, Style

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Recargar las variables de entorno
load_dotenv()

# Forzar recarga del módulo de configuración
importlib.reload(config_module)

# Crear una nueva instancia de configuración
config = config_module.Config()

# Validar configuración
config.validate_env()

class RedditIngestor:
    """Clase para ingestar datos desde Reddit API hacia la capa Bronze"""
    
    def __init__(self):
        # Inicializar cliente Reddit
        self.reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT
        )
        
        # Cargar esquema bronze
        with open(config.BRONZE_SCHEMA_PATH, 'r') as f:
            self.bronze_schema = json.load(f)
        
        self.bronze_fields = set(self.bronze_schema["properties"].keys())
        
        self.use_pushshift = config.USE_PUSHSHIFT  # Nueva configuración para Pushshift
        
    def extract_submission_data(self, submission) -> Dict[str, Any]:
        """Extrae y estructura datos de un submission de Reddit"""
        
        # Obtener datos raw
        submission_vars = vars(submission)
        raw_payload = {k: v for k, v in submission_vars.items() 
                      if isinstance(v, (str, int, float, bool, type(None)))}
        
        # Procesar campo 'edited' (puede ser bool o timestamp)
        edited_bool = False
        edited_utc = None
        if hasattr(submission, 'edited') and submission.edited:
            if isinstance(submission.edited, bool):
                edited_bool = submission.edited
            else:
                edited_bool = True
                edited_utc = float(submission.edited)
        
        # Timestamps
        ingest_ts = datetime.now(timezone.utc)
        created_ts = datetime.fromtimestamp(submission.created_utc, timezone.utc)
        
        # Datos core mapeados
        core_data = {
            "id": str(submission.id),
            "title": str(submission.title),
            "selftext": str(submission.selftext) if submission.selftext else "",
            "author": str(submission.author) if submission.author else None,
            "author_fullname": getattr(submission, 'author_fullname', None),
            "author_premium": getattr(submission, 'author_premium', False),
            "author_patreon_flair": getattr(submission, 'author_patreon_flair', False),
            "author_flair_text": getattr(submission, 'author_flair_text', None),
            "author_flair_css_class": getattr(submission, 'author_flair_css_class', None),
            "author_flair_type": getattr(submission, 'author_flair_type', None),
            "subreddit": str(submission.subreddit),
            "subreddit_id": str(submission.subreddit_id),
            "subreddit_name_prefixed": str(submission.subreddit_name_prefixed),
            "subreddit_type": getattr(submission, 'subreddit_type', None),
            "subreddit_subscribers": getattr(submission, 'subreddit_subscribers', 0),
            "created_utc": float(submission.created_utc),
            "created_ts": created_ts.isoformat(),
            "edited_bool": edited_bool,
            "edited_utc": edited_utc,
            "distinguished": getattr(submission, 'distinguished', None),
            "is_self": bool(submission.is_self),
            "over_18": bool(submission.over_18),
            "spoiler": bool(submission.spoiler),
            "locked": bool(submission.locked),
            "archived": bool(submission.archived),
            "quarantine": bool(getattr(submission, 'quarantine', False)),
            "stickied": bool(submission.stickied),
            "is_original_content": bool(getattr(submission, 'is_original_content', False)),
            "is_meta": bool(getattr(submission, 'is_meta', False)),
            "is_crosspostable": bool(getattr(submission, 'is_crosspostable', True)),
            "score": int(submission.score),
            "ups": int(submission.ups),
            "downs": int(getattr(submission, 'downs', 0)),
            "upvote_ratio": float(submission.upvote_ratio),
            "num_comments": int(submission.num_comments),
            "num_crossposts": int(getattr(submission, 'num_crossposts', 0)),
            "view_count": getattr(submission, 'view_count', None),
            "visited": bool(getattr(submission, 'visited', False)),
            "hide_score": bool(getattr(submission, 'hide_score', False)),
            "send_replies": bool(getattr(submission, 'send_replies', True)),
            "can_mod_post": bool(getattr(submission, 'can_mod_post', False)),
            "suggested_sort": getattr(submission, 'suggested_sort', None),
            "domain": str(submission.domain),
            "url": str(submission.url),
            "url_overridden_by_dest": getattr(submission, 'url_overridden_by_dest', None),
            "permalink": str(submission.permalink),
            "thumbnail": str(getattr(submission, 'thumbnail', '')),
            "thumbnail_height": getattr(submission, 'thumbnail_height', None),
            "thumbnail_width": getattr(submission, 'thumbnail_width', None),
            "post_hint": getattr(submission, 'post_hint', None),
            "is_video": bool(submission.is_video),
            "is_gallery": bool(getattr(submission, 'is_gallery', False)),
            "media_only": bool(getattr(submission, 'media_only', False)),
            "media": json.dumps(submission.media) if submission.media else None,
            "secure_media": json.dumps(getattr(submission, 'secure_media', None)) if getattr(submission, 'secure_media', None) else None,
            "media_embed": json.dumps(getattr(submission, 'media_embed', {})),
            "secure_media_embed": json.dumps(getattr(submission, 'secure_media_embed', {})),
            "preview": json.dumps(getattr(submission, 'preview', {})) if hasattr(submission, 'preview') else None,
            "gallery_data": json.dumps(getattr(submission, 'gallery_data', None)) if getattr(submission, 'gallery_data', None) else None,
            "media_metadata": json.dumps(getattr(submission, 'media_metadata', {})),
            "link_flair_text": getattr(submission, 'link_flair_text', None),
            "link_flair_css_class": getattr(submission, 'link_flair_css_class', None),
            "link_flair_type": getattr(submission, 'link_flair_type', None),
            "link_flair_richtext": json.dumps(getattr(submission, 'link_flair_richtext', [])),
            "author_flair_richtext": json.dumps(getattr(submission, 'author_flair_richtext', [])),
            "crosspost_parent": getattr(submission, 'crosspost_parent', None),
            "crosspost_parent_list": json.dumps(getattr(submission, 'crosspost_parent_list', [])),
            "parent_whitelist_status": getattr(submission, 'parent_whitelist_status', None),
            "whitelist_status": getattr(submission, 'whitelist_status', None),
            "wls": getattr(submission, 'wls', None),
            "pwls": getattr(submission, 'pwls', None),
            "content_categories": json.dumps(getattr(submission, 'content_categories', [])),
            "discussion_type": getattr(submission, 'discussion_type', None),
            "treatment_tags": json.dumps(getattr(submission, 'treatment_tags', [])),
            "is_robot_indexable": bool(getattr(submission, 'is_robot_indexable', True)),
            "author_is_blocked": bool(getattr(submission, 'author_is_blocked', False)),
            "awarders": json.dumps(getattr(submission, 'awarders', [])),
            "all_awardings": json.dumps(getattr(submission, 'all_awardings', [])),
            "total_awards_received": int(getattr(submission, 'total_awards_received', 0)),
            "gilded": int(submission.gilded),
            "gildings": json.dumps(getattr(submission, 'gildings', {})),
            "poll_data": json.dumps(getattr(submission, 'poll_data', None)) if getattr(submission, 'poll_data', None) else None,
            "_ingest_ts": ingest_ts.timestamp(),
            "_ingest_ts_ts": ingest_ts.isoformat(),
            "raw_payload": json.dumps(raw_payload)
        }
        
        # Campos extras (no mapeados)
        extras = {}
        for key, value in submission_vars.items():
            if key not in self.bronze_fields and isinstance(value, (str, int, float, bool)):
                extras[key] = str(value)
        
        core_data["extras"] = extras
        
        return core_data
    
    def fetch_pushshift_data(self, subreddit: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch historical data from Pushshift API."""
        url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&size={limit}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            logger.error(f"Pushshift API error for subreddit {subreddit}: {response.status_code}")
            return []

    def fetch_pushshift_data_with_retries(self, subreddit: str, limit: int, max_retries: int = 5, initial_wait: float = 1.0) -> List[Dict[str, Any]]:
        """Fetch historical data from Pushshift API with retries."""
        retries = 0
        wait_time = initial_wait

        while retries < max_retries:
            try:
                url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&size={limit}"
                response = requests.get(url)
                if response.status_code == 200:
                    return response.json().get("data", [])
                else:
                    logger.error(f"Pushshift API error for subreddit {subreddit}: {response.status_code}")
                    if response.status_code == 403:
                        logger.warning("Deshabilitando Pushshift globalmente debido a errores 403.")
                        self.use_pushshift = False  # Deshabilitar Pushshift globalmente
                        break  # No reintentar si es un error 403
            except Exception as e:
                logger.error(f"Error al conectar con Pushshift: {e}")

            retries += 1
            logger.warning(f"Reintentando Pushshift ({retries}/{max_retries}) en {wait_time:.2f} segundos...")
            time.sleep(wait_time)
            wait_time *= 2  # Incrementar el tiempo de espera exponencialmente

        logger.warning(f"Fallo Pushshift después de {max_retries} intentos. Cambiando a PRAW.")
        return []  # Retornar lista vacía si falla

    def handle_api_rate_limit(self):
        """Implementa un backoff exponencial para manejar errores 429."""
        backoff_time = random.uniform(1, 3)  # Tiempo inicial de espera
        max_backoff = 60  # Tiempo máximo de espera

        while True:
            try:
                yield  # Salir si la solicitud es exitosa
            except HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning(f"Rate limit alcanzado. Esperando {backoff_time:.2f} segundos...")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, max_backoff)  # Incrementar tiempo de espera
                else:
                    raise  # Re-lanzar otros errores

    def process_subreddit_with_fallback(self, subreddit_name):
        """Procesa un subreddit con Pushshift y fallback a PRAW."""
        try:
            logger.info(f"Procesando r/{subreddit_name}")
            submissions = []

            # Usar Pushshift solo si está habilitado
            if self.use_pushshift:
                submissions = self.fetch_pushshift_data_with_retries(subreddit_name, config.MICROBATCH_LIMIT)

            # Si Pushshift está deshabilitado o no devuelve resultados, usar PRAW
            if not self.use_pushshift or not submissions:
                logger.warning(f"Usando PRAW para r/{subreddit_name}")
                subreddit = self.reddit.subreddit(subreddit_name)
                submissions = list(subreddit.new(limit=config.MICROBATCH_LIMIT))

            return submissions
        except Exception as e:
            logger.error(f"Error procesando subreddit {subreddit_name}: {e}")
            return []

    def ingest_microbatch(self, show_progress: bool = False) -> None:
        """Ejecuta un microbatch de ingesta desde Reddit o Pushshift"""

        logger.info(f"Iniciando microbatch para subreddits: {', '.join(config.SUBREDDITS)}")

        # Validar que los subreddits no estén vacíos
        if not config.SUBREDDITS:
            logger.error("No se encontraron subreddits configurados para la ingesta.")
            return

        all_submissions = []

        # Medir el tiempo total de procesamiento
        start_time = time.time()

        # Registrar el número de posts procesados por subreddit
        subreddit_post_counts = {}

        def process_subreddit(subreddit_name):
            submissions = self.process_subreddit_with_fallback(subreddit_name)

            # Personalizar la barra de progreso para cada subreddit
            if show_progress:
                submissions = tqdm(
                    submissions,
                    desc=f"{Fore.GREEN}Procesando r/{subreddit_name}{Style.RESET_ALL}",
                    bar_format="{l_bar}{bar:30}{r_bar}"
                )

            subreddit_post_counts[subreddit_name] = 0

            for submission in submissions:
                try:
                    submission_data = self.extract_submission_data(submission)
                    all_submissions.append(submission_data)
                    subreddit_post_counts[subreddit_name] += 1
                except Exception as e:
                    logger.error(f"Error procesando submission: {e}")

        subreddit_list = config.SUBREDDITS

        # Personalizar la barra de progreso para subreddits
        if show_progress:
            subreddit_list = tqdm(
                subreddit_list,
                desc=f"{Fore.BLUE}Procesando subreddits{Style.RESET_ALL}",
                bar_format="{l_bar}{bar:30}{r_bar}"
            )

        with ThreadPoolExecutor(max_workers=config.THREADS) as executor:
            executor.map(process_subreddit, subreddit_list)

        # Registrar el tiempo total de procesamiento
        total_time = time.time() - start_time
        logger.info(f"Tiempo total de procesamiento: {total_time:.2f} segundos")

        # Registrar el número de posts procesados por subreddit
        for subreddit, count in subreddit_post_counts.items():
            logger.info(f"Subreddit: r/{subreddit}, Posts procesados: {count}")

        if all_submissions:
            self._save_to_partitioned_parquet(all_submissions)
            logger.info(f"Microbatch completado: {len(all_submissions)} submissions guardados")
            if show_progress:
                print(f"Total de nuevos posts agregados: {len(all_submissions)}")
        else:
            logger.warning("No se obtuvieron submissions en este microbatch")
    
    def _save_to_partitioned_parquet(self, submissions: List[Dict[str, Any]]) -> None:
        """Guarda submissions en formato Parquet particionado"""
        try:
            df = pl.DataFrame(submissions, infer_schema_length=1000)

            # Asegurar estructura de directorio particionada
            for row in df.iter_rows(named=True):
                date_partition = datetime.fromtimestamp(row["created_utc"]).strftime("%Y/%m/%d")
                subreddit_partition = row["subreddit"]
                partition_path = config.BRONZE_DIR / subreddit_partition / date_partition
                partition_path.mkdir(parents=True, exist_ok=True)

                # Guardar cada fila como un archivo Parquet
                file_path = partition_path / f"{row['id']}.parquet"
                pl.DataFrame([row]).write_parquet(file_path, compression="snappy")

            logger.info(f"Guardado {len(df)} submissions en formato Parquet particionado.")
        except Exception as e:
            logger.error(f"Error al guardar submissions en Parquet: {e}")
            for submission in submissions:
                logger.debug(f"Submission problemático: {submission}")

def main():
    """Función principal para ejecutar ingesta"""
    
    ingestor = RedditIngestor()
    
    # Determinar si se muestra la barra de progreso
    show_progress = os.getenv("SHOW_PROGRESS", "false").lower() == "true"

    # Ejecutar microbatch único o bucle continuo
    if config.MICROBATCH_INTERVAL_SEC > 0:
        logger.info(f"Iniciando ingesta continua cada {config.MICROBATCH_INTERVAL_SEC} segundos")
        while True:
            try:
                ingestor.ingest_microbatch(show_progress=show_progress)
                time.sleep(config.MICROBATCH_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("Ingesta detenida por usuario")
                break
            except Exception as e:
                logger.error(f"Error en ingesta: {e}")
                time.sleep(60)  # Esperar 1 minuto antes de reintentar
    else:
        # Ejecutar solo una vez
        ingestor.ingest_microbatch(show_progress=show_progress)

if __name__ == "__main__":
    main()
