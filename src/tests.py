#from scripts.retrieval import Retriever
from scripts.query_construction import finder_tmp
import json
#ret = Retriever()

x = finder_tmp(o='urw:urw_author_2')
print(x)

'''text = "Find all the books that have been written by Chinua Achebe"

x = ret.extract_knowledge(text)
print(x)
all_ents = ret.link_entities(json.loads(x))

print(all_ents)'''
