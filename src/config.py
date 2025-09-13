import os
from dotenv import load_dotenv
from pathlib import Path

# Cargar variables de entorno
load_dotenv()

class Config:
    """Configuración de la aplicación"""
    
    # Credenciales de Reddit
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "microbatches-reddit/1.0")
    
    # Configuración de microbatches
    SUBREDDITS = os.getenv("SUBREDDITS", "politics,worldnews,technology,health,ecuador,business,technews,artificial,futurology,tecnología,ecuador,datasciencees").split(",")
    MICROBATCH_LIMIT = int(os.getenv("MICROBATCH_LIMIT", "100"))
    MICROBATCH_INTERVAL_SEC = int(os.getenv("MICROBATCH_INTERVAL_SEC", "60"))
    
    # Rutas de datos
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    BRONZE_DIR = DATA_DIR / "bronze"
    SILVER_DIR = DATA_DIR / "silver"
    GOLD_DIR = DATA_DIR / "gold"
    REPORTS_DIR = DATA_DIR / "reports"
    MODELS_DIR = BASE_DIR / "models"
    SCHEMAS_DIR = BASE_DIR / "src" / "schemas"
    
    # Archivos de esquemas
    BRONZE_SCHEMA_PATH = SCHEMAS_DIR / "bronze_schema.json"
    SILVER_SCHEMA_PATH = SCHEMAS_DIR / "silver_schema.json"
    
    # Configuración de Pushshift
    USE_PUSHSHIFT = os.getenv("USE_PUSHSHIFT", "false").lower() == "true"
    
    # Configuración de hilos para procesamiento paralelo
    THREADS = int(os.getenv("THREADS", "12"))
    
    @classmethod
    def validate_env(cls):
        """Valida que las variables de entorno requeridas estén configuradas"""
        required_vars = [
            "REDDIT_CLIENT_ID",
            "REDDIT_CLIENT_SECRET"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Variables de entorno faltantes: {', '.join(missing_vars)}")
        
        return True

# Instancia global de configuración
config = Config()
