from collections import defaultdict

from eventvec.server.train.vectorizer.event_detector import EventDetector
from eventvec.server.model.document_models.document_model import Document
from eventvec.utils.spacy_utils.utils import SpacyUtils
from eventvec.server.train.vectorizer.dep_parser_model import parse_sentence
from eventvec.server.train.vectorizer.event_extractor import EventExtractor
from eventvec.server.train.vectorizer.event_relationship_extractor import EventRelationshipExtractor
from eventvec.server.train.vectorizer.coreference_resolver import CoreferenceResolver

class DocumentParser:
    def __init__(self):
        self._coreference_resolver = CoreferenceResolver()
        self._event_detector = EventDetector()
        self._event_extractor = EventExtractor()
        self._event_relationship_extractor = EventRelationshipExtractor()
        self._spacy_utils = SpacyUtils()


    def parse(self, document_text):
        document = Document()
        document.set_text(document_text)
        document = self._coreference_resolver.resolve(document)
        spacy_doc = self._spacy_utils.get_spacy_doc(document.resolved_text())
        for sentence in spacy_doc.sents:
            root, psentence = parse_sentence(sentence)
            detected_events = self._event_detector.detect(psentence)
            extracted_events = []
            for event in detected_events['verb_roots']:
                extracted_event = self._event_extractor.extract_verb_events(event, psentence)
                extracted_events.append(extracted_event)
                document.add_events(extracted_event)
            for event in detected_events['date_roots']:
                extracted_event = self._event_extractor.extract_date_events(event, psentence)
                extracted_events.append(extracted_event)
                document.add_events(extracted_event)
            for event_1i, event_1 in enumerate(extracted_events):
                for event_2i, event_2 in enumerate(extracted_events):
                    if event_1i < event_2i:
                        relationships =\
                             self._event_relationship_extractor.extract(
                                psentence, event_1, event_2
                            )
                        document.extend_relationships(relationships)
        return document
