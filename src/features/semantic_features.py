#!/usr/bin/env python3
"""
Generador de features semánticas desde RDF
Extrae características ontológicas para ML
"""

import pandas as pd
from rdflib import Graph, Namespace
from pathlib import Path
import logging
import pyarrow.parquet as pq

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Namespaces
RR = Namespace("http://example.org/reddit-ontology#")
RT = Namespace("http://example.org/reddit-topics#")
SCHEMA = Namespace("https://schema.org/")
BASE = Namespace("http://example.org/reddit/")

def extract_semantic_features():
    """
    Extrae features semánticas desde el grafo RDF
    """
    
    # Rutas
    project_root = Path(__file__).parent.parent.parent
    rdf_file = project_root / "data/gold/rdf/instances/posts/posts.ttl"
    ontology_file = project_root / "data/gold/rdf/ontology/reddit-ontology.ttl"
    output_dir = project_root / "data/gold/features"
    output_file = output_dir / "semantic-features.parquet"
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not rdf_file.exists():
        logger.error(f"Archivo RDF no encontrado: {rdf_file}")
        return False
    
    try:
        # Cargar grafo
        logger.info(f"Cargando grafo RDF desde {rdf_file}")
        g = Graph()
        g.parse(rdf_file, format="turtle")
        g.parse(ontology_file, format="turtle")
        logger.info(f"Cargados {len(g)} triples")
        
        # Consulta SPARQL para extraer features
        query = """
        PREFIX rr: <http://example.org/reddit-ontology#>
        PREFIX rt: <http://example.org/reddit-topics#>
        PREFIX schema: <https://schema.org/>
        PREFIX base: <http://example.org/reddit/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?reddit_id ?post_type ?topic ?score ?comment_count
        WHERE {
            ?post rdf:type rr:Post ;
                  rr:redditId ?reddit_id ;
                  rr:hasPostType ?post_type_class ;
                  rr:score ?score ;
                  schema:commentCount ?comment_count .
            
            # Extraer nombre del tipo de post
            BIND(REPLACE(STR(?post_type_class), ".*#", "") AS ?post_type)
            
            # Topic es opcional
            OPTIONAL {
                ?post rr:hasTopic ?topic_uri .
                BIND(REPLACE(STR(?topic_uri), ".*#", "") AS ?topic)
            }
        }
        ORDER BY ?reddit_id
        """
        
        logger.info("Ejecutando consulta SPARQL...")
        results = g.query(query)
        
        # Convertir resultados a DataFrame
        data = []
        for row in results:
            data.append({
                'reddit_id': str(row.reddit_id),
                'post_type': str(row.post_type),
                'topic': str(row.topic) if row.topic else None,
                'score': int(row.score),
                'comment_count': int(row.comment_count)
            })
        
        if not data:
            logger.error("No se encontraron resultados en la consulta")
            return False
        
        df = pd.DataFrame(data)
        logger.info(f"Extraídas {len(df)} features semánticas")
        
        # Mostrar estadísticas
        logger.info(f"Distribución de tipos de post:")
        logger.info(f"{df['post_type'].value_counts().to_string()}")
        
        logger.info(f"Distribución de temas:")
        logger.info(f"{df['topic'].value_counts(dropna=False).to_string()}")
        
        # Guardar como parquet
        logger.info(f"Guardando features en {output_file}")
        df.to_parquet(output_file, index=False)
        
        # Verificar archivo guardado
        saved_df = pd.read_parquet(output_file)
        logger.info(f"Verificación: archivo guardado con {len(saved_df)} filas y {len(saved_df.columns)} columnas")
        logger.info(f"Columnas: {list(saved_df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error extrayendo features semánticas: {e}")
        return False

def main():
    """Función principal"""
    logger.info("Iniciando extracción de features semánticas")
    
    success = extract_semantic_features()
    
    if success:
        logger.info("Extracción completada exitosamente")
        return 0
    else:
        logger.error("Extracción falló")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
