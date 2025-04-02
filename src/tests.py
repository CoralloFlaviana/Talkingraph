from scripts.retrieval import Retriever
import json
ret = Retriever()

text = """Find all the books that have been written by Chinua Achebe"""

x = ret.extract_knowledge(text)
print(x)
all_ents = ret.link_entities(json.loads(x))

print(all_ents)
