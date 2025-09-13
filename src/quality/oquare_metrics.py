#!/usr/bin/env python3
"""
Métricas de calidad OQuaRE (versión mejorada)
Calcula conteos y métricas más fieles de calidad ontológica
"""

from rdflib import Graph, Namespace, RDF, RDFS, OWL
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Namespaces
RR = Namespace("https://wfr.ai/onto/reddit#")
RT = Namespace("https://wfr.ai/onto/reddit#")  # Unificado

def load_graph_union():
    """
    Carga ontología e instancias en un grafo unificado
    """
    project_root = Path(__file__).parent.parent.parent
    ontology_file = project_root / "data/gold/rdf/ontology/reddit-ontology.ttl"
    topic_file = project_root / "data/gold/rdf/ontology/topic-scheme.ttl"
    instances_file = project_root / "data/gold/rdf/instances/posts/posts.ttl"
    
    g = Graph()
    
    # Cargar ontología
    if ontology_file.exists():
        logger.info(f"Cargando ontología desde {ontology_file}")
        g.parse(ontology_file, format="turtle")
    
    # Cargar esquema de temas
    if topic_file.exists():
        logger.info(f"Cargando esquema de temas desde {topic_file}")
        g.parse(topic_file, format="turtle")
    
    # Añadir instancias (sin fallar si no existen aún)
    if instances_file.exists():
        logger.info(f"Cargando instancias desde {instances_file}")
        g.parse(instances_file, format="turtle")
    else:
        logger.warning(f"Archivo de instancias no encontrado: {instances_file}")
    
    return g

def calculate_oquare_metrics():
    """
    Calcula métricas proxy de OQuaRE mejoradas
    
    Métricas implementadas:
    - RROnto: Relationships Richness (relaciones por entidad con densidad)  
    - AROnto: Attribute Richness (atributos por clase con anotaciones)
    - INROnto: Inheritance Richness (profundidad jerárquica)
    """
    
    try:
        # Cargar grafo unificado
        g = load_graph_union()
        
        logger.info("Calculando métricas OQuaRE mejoradas...")
        
        # ===== CONTEOS BÁSICOS =====
        
        # Contar clases (incluyendo SKOS concepts)
        classes_query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT (COUNT(DISTINCT ?class) AS ?count)
        WHERE {
            { ?class a owl:Class } UNION { ?class a skos:Concept }
        }
        """
        classes_count = int(list(g.query(classes_query))[0][0])
        
        # Contar propiedades objeto (incluyendo SKOS)
        object_props_query = """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT (COUNT(DISTINCT ?prop) AS ?count)
        WHERE {
            { ?prop a owl:ObjectProperty } UNION 
            { ?prop a skos:broader } UNION
            { ?s skos:broader ?o . BIND(skos:broader AS ?prop) }
        }
        """
        object_props_count = int(list(g.query(object_props_query))[0][0])
        
        # Contar propiedades de datos
        data_props_query = """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT (COUNT(DISTINCT ?prop) AS ?count)
        WHERE {
            ?prop a owl:DatatypeProperty .
        }
        """
        data_props_count = int(list(g.query(data_props_query))[0][0])
        
        # Contar propiedades de anotación (labels, comments)
        annotation_props_query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT (COUNT(DISTINCT ?prop) AS ?count)
        WHERE {
            ?prop a owl:AnnotationProperty .
            FILTER(?prop IN (rdfs:label, rdfs:comment, skos:prefLabel, skos:altLabel, skos:definition))
        }
        """
        annotation_props_result = list(g.query(annotation_props_query))
        annotation_props_count = int(annotation_props_result[0][0]) if annotation_props_result else 5  # Manual count
        
        # Contar triples totales
        total_triples = len(g)
        
        # Contar instancias
        instances_query = """
        PREFIX rr: <https://wfr.ai/onto/reddit#>
        SELECT (COUNT(?instance) AS ?count)
        WHERE {
            ?instance a rr:Post .
        }
        """
        instances_count = int(list(g.query(instances_query))[0][0])
        
        # Contar relaciones jerárquicas (subclass + skos:broader)
        subclass_query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT (COUNT(?relation) AS ?count)
        WHERE {
            { ?subclass rdfs:subClassOf ?superclass . BIND(?subclass AS ?relation) }
            UNION
            { ?narrower skos:broader ?broader . BIND(?narrower AS ?relation) }
        }
        """
        subclass_count = int(list(g.query(subclass_query))[0][0])
        
        # Contar relaciones reales en instancias
        instance_relations_query = """
        PREFIX rr: <https://wfr.ai/onto/reddit#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT (COUNT(?relation) AS ?count)
        WHERE {
            ?post ?relation ?target .
            ?relation a owl:ObjectProperty .
            ?post a rr:Post .
        }
        """
        instance_relations_result = list(g.query(instance_relations_query))
        instance_relations_count = int(instance_relations_result[0][0]) if instance_relations_result else 0
        
        # ===== MÉTRICAS OQuaRE MEJORADAS =====
        
        # RROnto: Relationships Richness mejorado
        # Densidad de relaciones considerando dominios y rangos
        total_possible_relations = object_props_count * classes_count if classes_count > 0 else 1
        actual_relations = object_props_count + subclass_count + (instance_relations_count / max(instances_count, 1))
        rr_onto = actual_relations / total_possible_relations if total_possible_relations > 0 else 0
        
        # AROnto: Attribute Richness mejorado 
        # Incluye propiedades de datos + anotaciones por clase
        total_props = data_props_count + annotation_props_count
        ar_onto = total_props / classes_count if classes_count > 0 else 0
        
        # INROnto: Inheritance Richness mejorado
        # Considera profundidad jerárquica efectiva
        max_possible_inheritance = classes_count - 1 if classes_count > 1 else 1
        inr_onto = subclass_count / max_possible_inheritance if max_possible_inheritance > 0 else 0
        
        # ===== RESULTADOS =====
        
        print("\n" + "="*60)
        print("MÉTRICAS OQuaRE MEJORADAS - ONTOLOGÍA REDDIT")
        print("="*60)
        
        print("\nCONTEOS BÁSICOS:")
        print(f"  Classes:                    {classes_count}")
        print(f"  Object Properties:          {object_props_count}")
        print(f"  Datatype Properties:        {data_props_count}")
        print(f"  Annotation Properties:      {annotation_props_count}")
        print(f"  Triples (Total):            {total_triples}")
        print(f"  Instances (Posts):          {instances_count}")
        print(f"  Subclass Relations:         {subclass_count}")
        print(f"  Instance Relations:         {instance_relations_count}")
        
        print("\nMÉTRICAS OQuaRE:")
        print(f"  RROnto (Relationships):     {rr_onto:.4f}")
        print(f"  AROnto (Attributes):        {ar_onto:.4f}")
        print(f"  INROnto (Inheritance):      {inr_onto:.4f}")
        
        print("\nINTERPRETACIÓN:")
        print(f"  - Densidad de relaciones: {rr_onto:.2%} del potencial máximo")
        print(f"  - Cada clase tiene en promedio {ar_onto:.1f} propiedades")
        print(f"  - {inr_onto:.1%} de la jerarquía posible está materializada")
        
        print("\n" + "="*60)
        
        logger.info("Métricas OQuaRE mejoradas calculadas exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error calculando métricas OQuaRE: {e}")
        return False

def main():
    """Función principal"""
    logger.info("Iniciando cálculo de métricas OQuaRE mejoradas")
    
    success = calculate_oquare_metrics()
    
    if success:
        logger.info("Cálculo completado exitosamente")
        return 0
    else:
        logger.error("Cálculo falló")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
