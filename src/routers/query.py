import requests
from fastapi import FastAPI, HTTPException, Query, APIRouter
from SPARQLWrapper import SPARQLWrapper, JSON
import yaml
import os,json
from internal.schemas import SearchResponse, FindResult, SearchResultURI
from internal.config import config as config 
from scripts.retrieval import Retriever
from scripts.query_construction import finder, searchExactly, searchRegex, searchTypeEntity, rel, explorationRel, finder_tmp

retriever = Retriever()


query = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)


# Configura endpoint 
SPARQL_ENDPOINT = config.endpoint


@query.get("/search_exactly", response_model=SearchResponse)
def serch_exactly(label: str, property: str=None) -> SearchResponse:
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    
    '''entity_key = f"entità{property}"  # Nome chiave da cercare
    
    # Controllo se esiste l'entità richiesta
    if entity_key not in config.namespace.right:
        raise HTTPException(status_code=400, detail=f"Entity {entity_key} not found in config")

    configEntity = config.namespace.right[entity_key].rel  

    # Controllo se il prefisso urw è disponibile
    urw_prefix = config.prefix["urw"]
    if not urw_prefix:
        raise HTTPException(status_code=500, detail="Prefix is missing in configuration")'''

    query = searchExactly(label,property)

    
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPARQL Query Error: {str(e)}")

    return {"results": results["results"]["bindings"]}
    


@query.get("/search_regex", response_model=SearchResponse)
def search_regex(label: str, property: str=None) -> SearchResponse:
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    
    '''entity_key = f"entità{numberEntity}"  # Nome chiave da cercare
    
    # Controllo se esiste l'entità richiesta
    if entity_key is not None:
        if entity_key not in config.namespace.right:
            raise HTTPException(status_code=400, detail=f"Entity {entity_key} not found in config")

        configEntity = config.namespace.right[entity_key].rel  

    # Controllo se il prefisso urw è disponibile
    urw_prefix = config.prefix["urw"]
    if not urw_prefix:
        raise HTTPException(status_code=500, detail="Prefix is missing in configuration")'''
    
    query=searchRegex(label, property)

    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPARQL Query Error: {str(e)}")

    return {"results": results["results"]["bindings"]}


    

@query.get("/find", response_model=FindResult)
def find(rel: str, o: str) -> FindResult:
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    
    # Controllo se il prefisso urw è disponibile
    urw_prefix = config.prefix["urw"]
    if not urw_prefix:
        raise HTTPException(status_code=500, detail="Prefix is missing in configuration")

    
    entity_key = f"entità{rel}"  # Nome chiave da cercare
    
    # Controllo se esiste l'entità richiesta
    if entity_key not in config.namespace.right:
        raise HTTPException(status_code=400, detail=f"Entity {entity_key} not found in config")

    configEntity = config.namespace.right[entity_key].rel  

    query=finder(urw_prefix, configEntity, o)

    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPARQL Query Error: {str(e)}")

    # Trasforma la risposta per Pydantic
    bindings = results["results"]["bindings"]
    formatted_results = [{"s": item["s"], "sogg": item["sogg"]} for item in bindings]

    return {"results": formatted_results}


@query.get("/search_typeEntity", response_model=SearchResultURI)
def search_type(entitytype: str) -> SearchResultURI:
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    
    # Controllo se il prefisso urw è disponibile
    urw_prefix = config.prefix["urw"]
    if not urw_prefix:
        raise HTTPException(status_code=500, detail="Prefix is missing in configuration")

    
    entity_key = f"entitytype{entitytype}"  # Nome chiave da cercare
    
    # Controllo se esiste l'entità richiesta
    if entity_key not in config.namespace.left:
        raise HTTPException(status_code=400, detail=f"Entity {entity_key} not found in config")

    prefix_type = config.namespace.left[entity_key].prefix
    entity_type = config.namespace.left[entity_key].type  

    query=searchTypeEntity(urw_prefix, entity_type)
    
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPARQL Query Error: {str(e)}")

    # Trasforma la risposta per Pydantic
    bindings = results["results"]["bindings"]
    formatted_results = [{"s": item["s"], "name": item["name"]} for item in bindings]

    return {"results": formatted_results}



@query.get("/graphrag")
def retrieve(text:str,type:str,k:int):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    template= eval(config.template)
    my_res = []
    result = retriever.extract_knowledge(template=template,text=text)
    result = json.loads(result)

    for res in result['entities'][type]:

        linked = retriever.link(res,type,k)
        
    
        my_res.extend(linked)

    query = finder_tmp(f"urw:{my_res[0][0]['entity']}")
    
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPARQL Query Error: {str(e)}")

    # Trasforma la risposta per Pydantic
    bindings = results["results"]["bindings"]
    formatted_results = [{"sogg": item["sogg"]} for item in bindings]

    return {"results": formatted_results}


@query.get("/rel")
def relTemp(ris: str) :
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    
    # Controllo se il prefisso urw è disponibile
    urw_prefix = config.prefix["urw"]
    if not urw_prefix:
        raise HTTPException(status_code=500, detail="Prefix is missing in configuration")
     
    ris="<"+ris+">"
    print(ris)
    query=rel(urw_prefix, ris)
    
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPARQL Query Error: {str(e)}")

    # Trasforma la risposta per Pydantic
    bindings = results["results"]["bindings"]
    formatted_results = [{"relazione": item["relazione"], "rel": item["rel"]} for item in bindings]

    return {"results": formatted_results}


@query.get("/entityFind", response_model=FindResult)
def entityFind(rel: str, o: str) -> FindResult:
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    
    # Controllo se il prefisso urw è disponibile
    urw_prefix = config.prefix["urw"]
    if not urw_prefix:
        raise HTTPException(status_code=500, detail="Prefix is missing in configuration")

    query=explorationRel(urw_prefix, rel, o)

    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPARQL Query Error: {str(e)}")

    # Trasforma la risposta per Pydantic
    bindings = results["results"]["bindings"]
    formatted_results = [{"s": item["s"], "sogg": item["sogg"]} for item in bindings]

    return {"results": formatted_results}