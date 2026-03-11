import json
from elasticsearch import Elasticsearch
from tqdm import tqdm

es = Elasticsearch("http://localhost:9200")

INDEX = "wikidata_entities"

mapping = {
    "mappings": {
        "properties": {
            "entity_id": {"type": "keyword"},
            "label": {"type": "text"},
            "aliases": {"type": "text"},
            "types": {"type": "keyword"},
            "neighbors": {"type": "keyword"}
        }
    }
}

es.indices.create(index=INDEX, body=mapping)

with open("data/raw/wikidata_dump.json") as f:

    for line in tqdm(f):

        doc = json.loads(line)

        es.index(index=INDEX, body=doc)