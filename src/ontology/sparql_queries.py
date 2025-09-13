#!/usr/bin/env python3
"""
SPARQL Competency Questions (CQs) para la ontología Reddit
Consultas principales para validar la completitud semántica
"""

from rdflib import Graph
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_rdf_data():
    """Carga los datos RDF (ontología + instancias)"""
    project_root = Path(__file__).parent.parent.parent
    
    # Archivos RDF
    ontology_file = project_root / "data/gold/rdf/ontology/reddit-ontology.ttl"
    topic_file = project_root / "data/gold/rdf/ontology/topic-scheme.ttl"
    instances_file = project_root / "data/gold/rdf/instances/posts/posts.ttl"
    
    # Crear grafo combinado
    graph = Graph()
    
    if ontology_file.exists():
        graph.parse(ontology_file, format="turtle")
        logger.info(f"Cargada ontología: {ontology_file}")
    
    if topic_file.exists():
        graph.parse(topic_file, format="turtle")
        logger.info(f"Cargado esquema de temas: {topic_file}")
    
    if instances_file.exists():
        graph.parse(instances_file, format="turtle")
        logger.info(f"Cargadas instancias: {instances_file}")
    
    return graph

def cq_missing_subreddit(graph):
    """Detecta posts sin rr:hasSubreddit"""
    query = """
    PREFIX rr: <https://wfr.ai/onto/reddit#>
    SELECT (COUNT(?p) AS ?n) WHERE {
      ?p a rr:Post .
      FILTER NOT EXISTS { ?p rr:hasSubreddit ?s . }
    }
    """
    results = list(graph.query(query))
    return int(results[0][0]) if results else 0

def cq_posts_with_subreddit_and_label(graph, limit=50):
    """Muestra posts con subreddit y sus labels"""
    query = f"""
    PREFIX rr: <https://wfr.ai/onto/reddit#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?p ?s (SAMPLE(?lbl) AS ?label) WHERE {{
      ?p a rr:Post ; rr:hasSubreddit ?s .
      OPTIONAL {{ ?s rdfs:label ?lbl }}
    }} GROUP BY ?p ?s
      LIMIT {int(limit)}
    """
    results = list(graph.query(query))
    return [(str(p).split('/')[-1], str(s), str(lbl) if lbl else None) for p, s, lbl in results]

def cq_technology_with_subreddit(graph, score_min=100):
    """Posts de Technology con subreddit explícito"""
    query = f"""
    PREFIX rr: <https://wfr.ai/onto/reddit#>
    SELECT ?p ?score ?s WHERE {{
      ?p a rr:Post ;
         rr:hasTopic rr:Technology ;
         rr:score ?score .
      OPTIONAL {{ ?p rr:hasSubreddit ?s }}
      FILTER(?score > {int(score_min)})
    }} ORDER BY DESC(?score)
      LIMIT 50
    """
    results = list(graph.query(query))
    return [(str(p).split('/')[-1], int(score), str(s) if s else None) for p, score, s in results]

def cq1_posts_by_subreddit(graph):
    """CQ1: ¿Cuántos posts hay por cada subreddit?"""
    query = """
    PREFIX rr: <https://wfr.ai/onto/reddit#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?subreddit (COUNT(?post) AS ?post_count)
    WHERE {
        ?post a rr:Post ;
              rr:hasSubreddit ?subreddit .
    }
    GROUP BY ?subreddit
    ORDER BY DESC(?post_count)
    """
    
    results = list(graph.query(query))
    subreddit_counts = []
    for row in results:
        subreddit = str(row[0])
        count = int(row[1])
        subreddit_counts.append((subreddit, count))
    
    return subreddit_counts

def cq2_high_score_tech_posts(graph):
    """CQ2: ¿Qué posts de Technology tienen score > 100?"""
    query = """
    PREFIX rr: <https://wfr.ai/onto/reddit#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?post ?score ?subreddit_label
    WHERE {
        ?post a rr:Post ;
              rr:hasTopic rr:Technology ;
              rr:score ?score ;
              rr:hasSubreddit ?subreddit .
        OPTIONAL { ?subreddit rdfs:label ?subreddit_label }
        FILTER(?score > 100)
    }
    ORDER BY DESC(?score)
    LIMIT 10
    """
    
    results = list(graph.query(query))
    tech_posts = []
    for row in results:
        post_id = str(row[0]).split('/')[-1] if '/' in str(row[0]) else str(row[0])
        score = int(row[1])
        # Extraer solo el nombre del subreddit del label r/subreddit
        subreddit_label = str(row[2]) if row[2] else "None"
        if subreddit_label.startswith("r/"):
            subreddit_name = subreddit_label[2:]  # Quitar "r/" prefix
        else:
            subreddit_name = subreddit_label
        tech_posts.append((post_id, score, subreddit_name))
    
    return tech_posts

def cq3_post_types_distribution(graph):
    """CQ3: ¿Cuál es la distribución de tipos de posts?"""
    query = """
    PREFIX rr: <https://wfr.ai/onto/reddit#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?post_type (COUNT(?post) AS ?count)
    WHERE {
        ?post a rr:Post ;
              rr:hasPostType ?post_type .
    }
    GROUP BY ?post_type
    ORDER BY DESC(?count)
    """
    
    results = list(graph.query(query))
    post_types = []
    
    for row in results:
        post_type = str(row[0]).split('#')[-1] if '#' in str(row[0]) else str(row[0])
        count = int(row[1])
        post_types.append((post_type, count))
    
    return post_types

def main():
    """Función principal - ejecuta todas las CQs"""
    print("\n" + "="*60)
    print("SPARQL COMPETENCY QUESTIONS - ONTOLOGÍA REDDIT")
    print("="*60)
    
    try:
        # Cargar datos RDF
        graph = load_rdf_data()
        total_triples = len(graph)
        print(f"\nTriples cargados: {total_triples}")
        
        if total_triples == 0:
            print("⚠️  No se encontraron datos RDF para consultar")
            return 1
        
        # Diagnóstico: Posts sin subreddit
        print("\n🔧 DIAGNÓSTICO: Posts sin subreddit")
        print("-" * 40)
        missing_count = cq_missing_subreddit(graph)
        print(f"Posts sin rr:hasSubreddit: {missing_count}")
        
        # Diagnóstico: Posts con subreddit y labels
        print("\n🔧 DIAGNÓSTICO: Muestra de posts con subreddit")
        print("-" * 50)
        sample_posts = cq_posts_with_subreddit_and_label(graph, 10)
        for post_id, subreddit_iri, label in sample_posts[:5]:
            print(f"  {post_id}: {subreddit_iri} -> {label}")
        
        # Diagnóstico: Technology posts con subreddit explícito
        print("\n🔧 DIAGNÓSTICO: Technology posts con subreddit")
        print("-" * 50)
        tech_posts = cq_technology_with_subreddit(graph, 100)
        for post_id, score, subreddit_iri in tech_posts[:5]:
            print(f"  {post_id}: score={score} subreddit_iri={subreddit_iri}")
        
        # CQ1: Posts por subreddit
        print("\n🔍 CQ1: Posts por subreddit")
        print("-" * 40)
        results1 = cq1_posts_by_subreddit(graph)
        for subreddit, count in results1:
            print(f"  {subreddit}: {count} posts")
        print(f"Total subreddits: {len(results1)}")
        
        # CQ2: Posts de Technology con score alto
        print("\n🔍 CQ2: Posts de Technology con score > 100")
        print("-" * 50)
        results2 = cq2_high_score_tech_posts(graph)
        for post_id, score, subreddit in results2:
            print(f"  {post_id}: score={score} (r/{subreddit})")
        print(f"Total posts encontrados: {len(results2)}")
        
        # CQ3: Distribución de tipos de posts
        print("\n🔍 CQ3: Distribución de tipos de posts")
        print("-" * 40)
        results3 = cq3_post_types_distribution(graph)
        total_posts = sum(count for _, count in results3)
        for post_type, count in results3:
            print(f"  {post_type}: {count} posts")
        print(f"Total posts analizados: {total_posts}")
        
        print("\n" + "="*60)
        print("✅ Todas las consultas SPARQL ejecutadas exitosamente")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error ejecutando consultas SPARQL: {e}")
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
