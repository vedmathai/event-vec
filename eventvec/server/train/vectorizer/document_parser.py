from collections import defaultdict

from eventvec.server.train.vectorizer.dep_parser_model import get_path
from eventvec.server.model.event_models.event_relationship_model import EventRelationship
from eventvec.server.train.vectorizer.event_detector import EventDetector
from eventvec.utils.timebank_prepositions import prep_to_relationships
from eventvec.server.model.document_models.document_model import Document
from eventvec.utils.spacy_utils.utils import get_spacy_doc
from eventvec.server.train.vectorizer.dep_parser_model import parse_sentence
from eventvec.server.train.vectorizer.event_extractor import EventExtractor
from eventvec.server.train.vectorizer.event_relationship_extractor import EventRelationshipExtractor



class DocumentParser:
    def __init__(self):
        self._event_detector = EventDetector()
        self._event_extractor = EventExtractor()
        self._event_relationship_extractor = EventRelationshipExtractor()


    def parse(self, document_text):
        document = Document()
        document.set_text(document_text)
        spacy_doc = get_spacy_doc(document_text)
        for sentence in spacy_doc.sents:
            root, psentence = parse_sentence(sentence)
            events = self._event_detector.detect(psentence)
            extracted_events = []
            for event in events:
                extracted_event = self._event_extractor.extract(event, psentence)
                extracted_events.append(extracted_event)
            for event_1i, event_1 in enumerate(extracted_events):
                for event_2i, event_2 in enumerate(extracted_events):
                    if event_1i < event_2i:
                        relationships =\
                             self._event_relationship_extractor.extract(
                                psentence, event_1, event_2
                            )
                        document.extend_relationships(relationships)
        return document
