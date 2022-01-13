class Document:
    def __init__(self):
        self._text = None
        self._resolved_text = None
        self._sentences = []
        self._events = []
        self._relationships = []

    def set_text(self, text):
        self._text = text

    def add_events(self, event):
        self._events.append(event)

    def add_sentences(self, sentence):
        self._sentences.append(sentence)    

    def set_resolved_text(self, resolved_text):
        self._resolved_text = resolved_text

    def text(self):
        return self._text

    def resolved_text(self):
        return self._resolved_text

    def events(self):
        return self._events

    def relationships(self):
        return self._relationships

    def extend_relationships(self, relationships):
        self._relationships.extend(relationships)

    @staticmethod
    def create(self, text):
        document = Document()
        document.set_text(text)
        return document