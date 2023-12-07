from typing import Any, Dict

from eventvec.server.data.factuality.factuality_datamodels.annotation import Annotation


class BeliefDatum():
    def __init__(self):
        super().__init__()
        self._tokens = []
        self._annotations = []
        self._event_string = None
        self._document = None
        self._sentence = None

    def tokens(self):
        return self._tokens
    
    def annotations(self):
        return self._annotations
    
    def text(self):
        return ' '.join(self.tokens())
    
    def event_string(self):
        return self._event_string
    
    def document(self):
        return self._document
    
    def sentence(self):
        return self._sentence
    
    def add_annotation(self, annotation):
        self._annotations.append(annotation)

    def set_tokens(self, tokens):
        self._tokens = tokens

    def set_text(self, text):
        self._text = text

    def set_event_string(self, event_string):
        self._event_string = event_string

    def set_document(self, document):
        self._document = document
    
    def set_sentence(self, sentence):
        self._sentence = sentence


    def to_dict(self):
        return {
            'tokens': self.tokens(),
            'annotations': [a.to_json() for a in self.annotations()],
            'event_string': self.event_string(),
            'document': self.document(),
            'sentence': self.sentence(),
        }

    @staticmethod
    def from_dict(dict_data: Dict[str, Any]):
        belief_datum = BeliefDatum()
        belief_datum.set_tokens(dict_data['tokens'])
        for annotation in dict_data['annotations']:
            belief_datum.add_annotation(annotation)
        belief_datum.set_event_string(dict_data['event_string'])
        belief_datum.set_document(dict_data['document'])
        belief_datum.set_sentence(dict_data['sentence'])
        return belief_datum
    
    @staticmethod
    def parse_raw_data(raw_data):
        belief_datum = BeliefDatum()
        belief_datum.set_tokens(raw_data['data']['tokens'])
        belief_datum.set_event_string(raw_data['data']['event_string'])
        belief_datum.set_sentence(raw_data['data']['sentence'])
        belief_datum.set_document(raw_data['data']['document'])
        for annotation_raw in raw_data['results']['judgments']:
            annotation = Annotation.parse_raw_data(annotation_raw)
            belief_datum.add_annotation(annotation)
        return belief_datum
