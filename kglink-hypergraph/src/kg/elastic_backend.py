from elasticsearch import Elasticsearch

class ElasticBackend:

    def __init__(self, host, index):
        self.es = Elasticsearch(host)
        self.index = index

    def search(self, mention, top_k=10):

        query = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": mention,
                    "fields": ["label^3", "aliases^2"]
                }
            }
        }

        res = self.es.search(index=self.index, body=query)

        entities = []

        for hit in res["hits"]["hits"]:
            src = hit["_source"]

            entities.append({
                "entity_id": src["entity_id"],
                "label": src["label"],
                "score": hit["_score"],
                "types": src.get("types", []),
                "neighbors": src.get("neighbors", [])
            })

        return entities