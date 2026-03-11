from .elastic_backend import ElasticBackend

class EntityLinker:

    def __init__(self, backend):
        self.backend = backend

    def link_cell(self, value):

        if not value:
            return []

        return self.backend.search(value)