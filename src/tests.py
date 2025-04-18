from scripts.retrieval import Retriever
import json
ret = Retriever()


x = ret.link_to_triples([{'entity':'urw:urw_author_1584','label':'Chinua Achebe'}])

print(x)
'''x = finder_tmp(o='urw:urw_author_2')
print(x)'''

'''text = "Trova tutte le pubblicazioni di Rossana Damiano"

x = ret.extract_knowledge(text)
print(x)
all_ents = ret.link_entities(json.loads(x))

print(all_ents)'''
