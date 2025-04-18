from transformers import AutoModelForCausalLM, AutoTokenizer
import torch,json
from sentence_transformers import SentenceTransformer
import json
import pandas as pd
import faiss
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from scripts.query_construction import finder_tmp
from internal.config import config as config 
from typing import Dict, Tuple, List

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

class Retriever:
    def __init__(self):
        self.model_name = "numind/NuExtract-tiny-v1.5"
        self.device = "mps"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.df = pd.read_parquet('data/entities_def.parquet')


    def extract_knowledge(self,text,template=config.template, max_length=10_000, max_new_tokens=4_000):
        model = self.model
        tokenizer = self.tokenizer
        
        prompt = f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>"""
        outputs = []
        with torch.no_grad():
            
            batch_encodings = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(model.device)
            pred_ids = model.generate(**batch_encodings, max_new_tokens=max_new_tokens)
            outputs += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        
        
        return outputs[0].split("<|output|>")[1]
    
    def link(self, entity, type,k):
        
        index = faiss.read_index(config.files[type])
        query_vector = self.sentence_model.encode([entity], return_tensors=True)
        
        distances, indices = index.search(query_vector, k)
        retrieved = list()
        
        for i,idx in enumerate(indices[0]):
            ent = self.df[(self.df.faiss_id==idx)&(self.df.type==type)]
            try:
                retrieved.append({'entity':ent.entity.values[0],'label':ent.label.values[0],'distance':distances[0][i].item()})
            except:
                continue
        return retrieved
    
    def link_entities(self, output_template,k):
        all_entities = list()
        for item in output_template['entities']:
            
            if len(output_template['entities'][item])>0:
                for ent in output_template['entities'][item]:
                    
                    processed = self.link(ent,item,k)
                    all_entities.extend(processed)
        
        all_entities = sorted(all_entities, key=lambda x: x['distance'])

        return all_entities
    
    def retrieve_triples(self, entity=Dict,property:Dict=None):
        sparql = SPARQLWrapper(config.endpoint)
        prop = next(iter(property))
        query = finder_tmp(entity["entity"],prop) if property is not None else finder_tmp(entity['uri'])
        print(query)

        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

        except Exception as e:
            raise Exception(f"SPARQL Query Error: {str(e)}")
        
        results = [(entity['label'],property[prop],x['sogg']['value']) for x in results['results']['bindings']]

        return results
    
    def link_to_triples(self, entities:List[Dict],k=1,properties=config.properties):
        all_triples = list()
        for item in entities[:k]:
            for prop in properties:
                
                retrieved = self.retrieve_triples(item,prop)
                
                all_triples.extend(retrieved)
        
        return all_triples





'''text = """Find books by authors similar to one written by Achebe"""

template = {
    "QuestionFocus": "",
    "entities": {
        "work": [],
        "person": [],
        "subject":[],
        "publisher":[]
    }
}

retr = Retriever()
res = json.loads(retr.extract_knowledge(template=template, text=text))

print(res)

for item in res['entities']['person']:
    print(retr.link(item,'person'))'''