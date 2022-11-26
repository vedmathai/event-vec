class EventTrigger:
    def __init__(self):
        self._id = None
        self._word = None
        self._pos = None
        self._relationships = {}
        self._visited = False
        self._source_documents = set()
        self._count = 0

    def id(self):
        return self._word.lower()

    def relationships(self):
        return self._relationships

    def relationship(self, relationship_id):
        return self._relationships[relationship_id]

    def word(self):
        return self._word

    def pos(self):
        return self._pos

    def visited(self):
        return self._visited

    def count(self):
        return self._count

    def reset(self):
        self._visited = False

    def set_id(self, id):
        self._id = id

    def set_word(self, word):
        self._word = word

    def add_relationship(self, relationship):
        self._relationships[relationship.id()] = relationship

    def set_pos(self, pos):
        self._pos = pos

    def set_visited_true(self):
        self._visited = True

    def set_visited_false(self):
        self._visited = False

    def add_source_document(self, source_document):
        self._source_documents.add(source_document)

    def increment_count(self):
        self._count += 1

    def source_documents(self):
        return self._source_documents
