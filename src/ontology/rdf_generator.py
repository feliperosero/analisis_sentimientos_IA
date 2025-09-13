#!/usr/bin/env python3
"""
Generador RDF para instancias de Reddit
Mapeo OBDA desde data/silver/silver.parquet a RDF
"""

import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
from rdflib.namespace import XSD, DCTERMS, PROV
from pathlib import Path
import logging
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Namespaces
RR = Namespace("https://wfr.ai/onto/reddit#")
RT = Namespace("https://wfr.ai/onto/reddit#")  # Unificado con RR
SCHEMA = Namespace("https://schema.org/")
BASE = Namespace("http://example.org/reddit/")

# --- Normalización segura de subreddit ---
def _norm_subreddit(name: str) -> str:
    """Normaliza nombres de subreddit"""
    if not name:
        return None
    s = str(name).strip()
    # Normaliza capitalización y variantes típicas
    mapping = {
        "Economics": "economics",
        "Health": "health", 
        "WorldNews": "worldnews",
        "Technology": "technology",
        "TechNews": "technews",
        "Business": "business",
        "Politics": "politics",
    }
    return mapping.get(s, s.lower())

def create_subreddit(g, base_ns, subreddit_name: str):
    """Crea individuo Subreddit con label"""
    sname = _norm_subreddit(subreddit_name)
    if not sname:
        return None
    s_iri = URIRef(f"{base_ns}r/{sname}")
    g.add((s_iri, RDF.type, RR.Subreddit))
    # etiqueta legible (útil para UI/queries)
    g.add((s_iri, RDFS.label, Literal(f"r/{sname}", lang="en")))
    return s_iri

def attach_subreddit(g, post_iri, base_ns, subreddit_name):
    """Adjunta subreddit a post con normalización"""
    s_iri = create_subreddit(g, base_ns, subreddit_name)
    if s_iri:
        g.add((post_iri, RR.hasSubreddit, s_iri))
    else:
        # Marca auditoría mínima (si quieres, comenta esta línea en producción)
        pass  # sin subreddit; revisarlo con SPARQL de diagnóstico

# Mapeo de temas completo (10 conceptos SKOS)
TOPIC_BY_SUBREDDIT = {
    "technology": "Technology",
    "technews": "TechNews", 
    "machinelearning": "MachineLearning",
    "artificialintelligence": "ArtificialIntelligence",
    "worldnews": "WorldNews",
    "science": "Science",
    "politics": "Politics",
    "business": "Business",
    "economics": "Economics",
    "health": "Health",
}

def attach_topic_by_subreddit(g, post_iri, subreddit_name):
    """Adjunta tema basado en subreddit normalizado"""
    sname = _norm_subreddit(subreddit_name)
    tlocal = TOPIC_BY_SUBREDDIT.get(sname)
    if tlocal:
        # Usar prefijo rr: unificado
        g.add((post_iri, RR.hasTopic, URIRef(f"https://wfr.ai/onto/reddit#{tlocal}")))

def clean_string(s):
    """Limpia strings para URIs"""
    if pd.isna(s) or s is None:
        return None
    return str(s).replace(" ", "_").replace("/", "_")

def map_post_type(row):
    """Mapea tipo de post según reglas de inferencia"""
    if row.get('es_texto', False):
        return RR.TextPost
    elif row.get('es_video', False):
        return RR.VideoPost
    elif row.get('es_galeria', False):
        return RR.GalleryPost
    elif row.get('pista_contenido') == 'link' and not row.get('es_video', False):
        return RR.LinkPost
    else:
        return RR.TextPost  # Default

def map_topic(subreddit_name):
    """Mapea subreddit a tema - DEPRECATED, usar attach_topic_by_subreddit"""
    # Mantenido para compatibilidad, pero se prefiere attach_topic_by_subreddit
    if not subreddit_name:
        return None
    
    subreddit_lower = subreddit_name.lower()
    
    if subreddit_lower in ['technology', 'technews', 'machinelearning', 'programming']:
        return URIRef("https://wfr.ai/onto/reddit#Technology")
    elif subreddit_lower in ['worldnews']:
        return URIRef("https://wfr.ai/onto/reddit#WorldNews")
    elif subreddit_lower in ['science', 'health']:
        return URIRef("https://wfr.ai/onto/reddit#Science")
    elif subreddit_lower in ['politics']:
        return URIRef("https://wfr.ai/onto/reddit#Politics") 
    elif subreddit_lower in ['business', 'economics']:
        return URIRef("https://wfr.ai/onto/reddit#Economics")
    else:
        return None

def generate_rdf_from_parquet(input_path, output_dir):
    """
    Genera RDF desde el archivo parquet de silver
    
    Args:
        input_path: Ruta al archivo silver.parquet
        output_dir: Directorio de salida para archivos RDF
    """
    
    # Crear directorios de salida
    output_dir = Path(output_dir)
    instances_dir = output_dir / "instances"
    posts_dir = instances_dir / "posts"
    subreddits_dir = instances_dir / "subreddits"
    authors_dir = instances_dir / "authors"
    domains_dir = instances_dir / "domains"
    
    for dir_path in [posts_dir, subreddits_dir, authors_dir, domains_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Leer datos
    logger.info(f"Leyendo datos desde {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Cargados {len(df)} registros")
    
    # Crear grafo principal
    g = Graph()
    g.bind("rr", RR)
    g.bind("rt", RT)
    g.bind("schema", SCHEMA)
    g.bind("dct", DCTERMS)
    g.bind("prov", PROV)
    
    # Contadores para métricas
    posts_count = 0
    subreddits_set = set()
    authors_set = set()
    domains_set = set()
    posts_with_topic = 0
    posts_with_type = 0
    
    # Procesar cada post
    logger.info("Generando triples RDF...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Procesando posts"):
        
        # IRI del post: /post/{id}
        post_id = clean_string(row.get('id'))
        if not post_id:
            continue
            
        post_uri = BASE[f"post/{post_id}"]
        posts_count += 1
        
        # Tipo básico
        g.add((post_uri, RDF.type, RR.Post))
        
        # redditId
        g.add((post_uri, RR.redditId, Literal(post_id, datatype=XSD.string)))
        
        # Título
        title = row.get('titulo')
        if title and not pd.isna(title):
            g.add((post_uri, DCTERMS.title, Literal(str(title))))
        
        # Texto del post
        selftext = row.get('texto')
        if selftext and not pd.isna(selftext):
            g.add((post_uri, SCHEMA.text, Literal(str(selftext))))
        
        # Timestamp
        created_utc = row.get('fecha_creacion_utc')
        if created_utc and not pd.isna(created_utc):
            # Convertir timestamp Unix a ISO 8601
            from datetime import datetime
            dt = datetime.fromtimestamp(created_utc)
            iso_datetime = dt.isoformat()
            g.add((post_uri, PROV.generatedAtTime, Literal(iso_datetime, datatype=XSD.dateTime)))
        
        # Score
        score = row.get('puntaje')
        if score is not None and not pd.isna(score):
            g.add((post_uri, RR.score, Literal(int(score), datatype=XSD.integer)))
        
        # Número de comentarios
        num_comments = row.get('total_comentarios')
        if num_comments is not None and not pd.isna(num_comments):
            g.add((post_uri, SCHEMA.commentCount, Literal(int(num_comments), datatype=XSD.integer)))
        
        # URL
        url = row.get('url')
        if url and not pd.isna(url):
            g.add((post_uri, SCHEMA.url, Literal(str(url))))
        
        # Subreddit: normalizado con attach_subreddit
        subreddit = row.get('subreddit_nombre')
        if subreddit:
            attach_subreddit(g, post_uri, BASE, subreddit)
            subreddits_set.add(_norm_subreddit(subreddit) or subreddit)
        
        # Autor: /u/{autor}
        author = clean_string(row.get('autor'))
        if author and author != '[deleted]':
            author_uri = BASE[f"u/{author}"]
            g.add((author_uri, RDF.type, RR.Author))
            g.add((post_uri, RR.hasAuthor, author_uri))
            authors_set.add(author)
        
        # Dominio: /domain/{dominio}
        domain = clean_string(row.get('dominio'))
        if domain:
            domain_uri = BASE[f"domain/{domain}"]
            g.add((domain_uri, RDF.type, RR.Domain))
            g.add((post_uri, RR.linksToDomain, domain_uri))
            domains_set.add(domain)
        
        # Inferir tipo de post
        post_type = map_post_type(row)
        g.add((post_uri, RDF.type, post_type))
        g.add((post_uri, RR.hasPostType, post_type))
        posts_with_type += 1
        
        # Inferir tema con nuevo mapeo
        attach_topic_by_subreddit(g, post_uri, subreddit)
        if subreddit and _norm_subreddit(subreddit) in TOPIC_BY_SUBREDDIT:
            posts_with_topic += 1
    
    # Guardar grafo principal
    output_file = posts_dir / "posts.ttl"
    logger.info(f"Guardando {len(g)} triples en {output_file}")
    g.serialize(destination=output_file, format="turtle")
    
    # Métricas
    logger.info(f"Generación completada:")
    logger.info(f"  Posts procesados: {posts_count}")
    logger.info(f"  Subreddits únicos: {len(subreddits_set)}")
    logger.info(f"  Autores únicos: {len(authors_set)}")
    logger.info(f"  Dominios únicos: {len(domains_set)}")
    logger.info(f"  Posts con tema: {posts_with_topic} ({posts_with_topic/posts_count*100:.1f}%)")
    logger.info(f"  Posts con tipo: {posts_with_type} ({posts_with_type/posts_count*100:.1f}%)")
    logger.info(f"  Total triples: {len(g)}")
    
    return {
        'posts_count': posts_count,
        'subreddits_count': len(subreddits_set),
        'authors_count': len(authors_set),
        'domains_count': len(domains_set),
        'posts_with_topic': posts_with_topic,
        'posts_with_type': posts_with_type,
        'total_triples': len(g),
        'output_file': str(output_file)
    }

def main():
    """Función principal"""
    import sys
    
    # Rutas desde raíz del proyecto
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data/silver/silver.parquet"
    output_dir = project_root / "data/gold/rdf"
    
    if not input_path.exists():
        logger.error(f"Archivo de entrada no encontrado: {input_path}")
        sys.exit(1)
    
    try:
        stats = generate_rdf_from_parquet(input_path, output_dir)
        logger.info("Proceso completado exitosamente")
        return stats
    except Exception as e:
        logger.error(f"Error en la generación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
