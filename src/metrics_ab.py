import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime, timedelta

from .config import config

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsComparator:
    """Compara métricas entre predicciones A y B"""
    
    def __init__(self):
        self.pred_a_path = config.REPORTS_DIR / "predicciones_a.parquet"
        self.pred_b_path = config.REPORTS_DIR / "predicciones_b.parquet"
    
    def compare_predictions(self) -> None:
        """Compara predicciones A y B y genera métricas"""
        
        logger.info("Iniciando comparación de métricas A vs B")
        
        # Cargar predicciones
        if not self.pred_a_path.exists():
            logger.error(f"Predicciones A no encontradas: {self.pred_a_path}")
            return
        
        if not self.pred_b_path.exists():
            logger.error(f"Predicciones B no encontradas: {self.pred_b_path}")
            return
        
        pred_a = pl.read_parquet(self.pred_a_path)
        pred_b = pl.read_parquet(self.pred_b_path)
        
        logger.info(f"Cargadas {len(pred_a)} predicciones A y {len(pred_b)} predicciones B")
        
        # Unir predicciones por ID
        combined = pred_a.join(
            pred_b,
            on=["id", "fecha_creacion", "subreddit_nombre"],
            how="inner",
            suffix="_b"
        ).rename({
            "prediccion": "prediccion_a",
            "probabilidad": "probabilidad_a",
            "modelo": "modelo_a",
            "ts_prediccion": "ts_prediccion_a"
        })
        
        logger.info(f"Registros combinados: {len(combined)}")
        
        if len(combined) == 0:
            logger.warning("No se pudieron combinar predicciones A y B")
            return
        
        # Calcular métricas
        metrics = self._calculate_metrics(combined)
        
        # Guardar métricas
        self._save_metrics(metrics)
        
        # Generar gráficos
        self._generate_plots(combined)
        
        logger.info("Comparación de métricas completada")
    
    def _calculate_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calcula métricas de comparación entre A y B"""
        
        # Métricas por ventana temporal (última hora, por ejemplo)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        
        # Métricas globales
        global_metrics = pl.DataFrame({
            "ventana": ["global"],
            "total_registros": [len(df)],
            "predicciones_a_positivas": [df.filter(pl.col("prediccion_a") == 1).height],
            "predicciones_b_positivas": [df.filter(pl.col("prediccion_b") == 1).height],
            "concordancia_ab": [df.filter(pl.col("prediccion_a") == pl.col("prediccion_b")).height],
            "diferencia_prob_media": [df.select((pl.col("probabilidad_b") - pl.col("probabilidad_a")).mean()).item()],
            "prob_a_media": [df.select(pl.col("probabilidad_a").mean()).item()],
            "prob_b_media": [df.select(pl.col("probabilidad_b").mean()).item()],
            "timestamp_calculo": [datetime.now().isoformat()]
        }).cast({
            "total_registros": pl.Int64,
            "predicciones_a_positivas": pl.Int64, 
            "predicciones_b_positivas": pl.Int64,
            "concordancia_ab": pl.Int64
        })
        
        # Métricas por subreddit
        subreddit_metrics = df.group_by("subreddit_nombre").agg([
            pl.len().alias("total_registros"),
            (pl.col("prediccion_a") == 1).sum().alias("predicciones_a_positivas"),
            (pl.col("prediccion_b") == 1).sum().alias("predicciones_b_positivas"),
            (pl.col("prediccion_a") == pl.col("prediccion_b")).sum().alias("concordancia_ab"),
            (pl.col("probabilidad_b") - pl.col("probabilidad_a")).mean().alias("diferencia_prob_media"),
            pl.col("probabilidad_a").mean().alias("prob_a_media"),
            pl.col("probabilidad_b").mean().alias("prob_b_media")
        ]).cast({
            "total_registros": pl.Int64,
            "predicciones_a_positivas": pl.Int64,
            "predicciones_b_positivas": pl.Int64,
            "concordancia_ab": pl.Int64
        }).with_columns([
            pl.col("subreddit_nombre").alias("ventana"),
            pl.lit(datetime.now().isoformat()).alias("timestamp_calculo")
        ]).select([
            "ventana", "total_registros", "predicciones_a_positivas", "predicciones_b_positivas",
            "concordancia_ab", "diferencia_prob_media", "prob_a_media", "prob_b_media", "timestamp_calculo"
        ])
        
        # Combinar métricas
        all_metrics = pl.concat([global_metrics, subreddit_metrics])
        
        return all_metrics
    
    def _save_metrics(self, metrics: pl.DataFrame) -> None:
        """Guarda métricas en archivo"""
        
        # Asegurar que el directorio existe
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Guardar métricas
        filepath = config.REPORTS_DIR / "metricas_ab.parquet"
        metrics.write_parquet(filepath)
        
        logger.info(f"Métricas guardadas en {filepath}")
        
        # También guardar en CSV para fácil lectura
        csv_filepath = config.REPORTS_DIR / "metricas_ab.csv"
        metrics.write_csv(csv_filepath)
        
        logger.info(f"Métricas también guardadas en {csv_filepath}")
    
    def _generate_plots(self, df: pl.DataFrame) -> None:
        """Genera gráficos de comparación"""
        
        try:
            # Configurar matplotlib
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Comparación de Predicciones A vs B', fontsize=16)
            
            # Convertir a pandas para plotting
            df_pd = df.to_pandas()
            
            # Gráfico 1: Distribución de probabilidades
            axes[0, 0].hist(df_pd['probabilidad_a'], alpha=0.7, label='Modelo A', bins=20)
            axes[0, 0].hist(df_pd['probabilidad_b'], alpha=0.7, label='Modelo B', bins=20)
            axes[0, 0].set_xlabel('Probabilidad')
            axes[0, 0].set_ylabel('Frecuencia')
            axes[0, 0].set_title('Distribución de Probabilidades')
            axes[0, 0].legend()
            
            # Gráfico 2: Scatter plot A vs B
            axes[0, 1].scatter(df_pd['probabilidad_a'], df_pd['probabilidad_b'], alpha=0.6)
            axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Línea de igualdad')
            axes[0, 1].set_xlabel('Probabilidad A')
            axes[0, 1].set_ylabel('Probabilidad B')
            axes[0, 1].set_title('Probabilidades A vs B')
            axes[0, 1].legend()
            
            # Gráfico 3: Concordancia por subreddit
            concordancia_por_sub = df.group_by("subreddit_nombre").agg([
                pl.len().alias("total"),
                (pl.col("prediccion_a") == pl.col("prediccion_b")).sum().alias("concordantes")
            ]).with_columns([
                (pl.col("concordantes") / pl.col("total") * 100).alias("porcentaje_concordancia")
            ])
            
            concordancia_pd = concordancia_por_sub.to_pandas()
            axes[1, 0].bar(concordancia_pd['subreddit_nombre'], concordancia_pd['porcentaje_concordancia'])
            axes[1, 0].set_xlabel('Subreddit')
            axes[1, 0].set_ylabel('% Concordancia')
            axes[1, 0].set_title('Concordancia entre A y B por Subreddit')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Gráfico 4: Diferencia de probabilidades
            diff_prob = df_pd['probabilidad_b'] - df_pd['probabilidad_a']
            axes[1, 1].hist(diff_prob, bins=20, alpha=0.7)
            axes[1, 1].axvline(diff_prob.mean(), color='red', linestyle='--', 
                              label=f'Media: {diff_prob.mean():.3f}')
            axes[1, 1].set_xlabel('Diferencia (B - A)')
            axes[1, 1].set_ylabel('Frecuencia')
            axes[1, 1].set_title('Distribución de Diferencias B - A')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Guardar figura
            plot_path = config.REPORTS_DIR / "comparacion_ab.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Gráfico guardado en {plot_path}")
            
        except Exception as e:
            logger.error(f"Error generando gráficos: {e}")

def main():
    """Función principal para ejecutar comparación de métricas"""
    
    comparator = MetricsComparator()
    comparator.compare_predictions()

if __name__ == "__main__":
    main()
