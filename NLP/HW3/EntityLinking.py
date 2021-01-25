import json
import math

amr_knowledge_graph = {
    "Romney": ["McDonnell", "Johnson"],
    "McDonnell": ["Romney", "Johnson"],
    "Johnson": ["McDonnell", "Romney"]
}

mcdonnell_article = "https://en.wikipedia.org/wiki/Bob_McDonnell"
kb = json.load(open('C:\\Users\\Karan Sarkar\\Google Drive\\RPI\\NLP\\kb.json'))
knowledge_base = {}
for kbid in kb.keys():
    knowledge_base[kbid] = set(kb[kbid]['links'])
keys = list(kb.keys())

def jaccard_sim(romney_article, johnson_article):
    romney_links = set([johnson_article, mcdonnell_article])
    johnson_links = set([romney_article, mcdonnell_article])
    romney_sim = len(knowledge_base[romney_article] & romney_links) / (len(knowledge_base[romney_article] | romney_links))
    johnson_sim = len(knowledge_base[johnson_article] & johnson_links) / (len(knowledge_base[johnson_article] | johnson_links))
    return romney_sim + johnson_sim

def log_jaccard_sim(romney_article, johnson_article):
    romney_links = set([johnson_article, mcdonnell_article])
    johnson_links = set([romney_article, mcdonnell_article])
    romney_sim = (len(knowledge_base[romney_article] & romney_links)) / math.log(len(knowledge_base[romney_article] | romney_links))
    johnson_sim = (len(knowledge_base[johnson_article] & johnson_links)) / math.log(len(knowledge_base[johnson_article] | johnson_links))
    return romney_sim + johnson_sim

sols = [(keys[i], keys[j + 10], log_jaccard_sim(keys[i], keys[j + 10])) for i in range(10) for j in range(10)]
sols.sort(key= lambda sol: sol[2], reverse = True)

for sol in sols:
    print(sol)

        
    
    