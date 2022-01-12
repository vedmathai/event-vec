from collections import defaultdict

from eventvec.server.train.vectorizer.dep_parser_model import get_path
from eventvec.server.model.event_models.event_relationship_model import EventRelationship
from eventvec.utils.timebank_prepositions import prep_to_relationships

class EventRelationshipExtractor:
    def extract(self, sentence, event_1, event_2):
        relationships = []
        node_1 = event_1.root_node()
        node_2 = event_2.root_node()
        if node_1.pos() in ['VERB'] and node_2.pos() in ['VERB']:
            dep = get_path(sentence, node_1.i(), node_2.i())
            dep_tup = []
            for ii, i in enumerate(dep):
                if i.dep() == 'prep':
                    dep_tup += [i.lemma()]
            dep_tup = '|'.join(dep_tup)
            if len(dep_tup) > 0 and len(dep) > 0:
                for rel in prep_to_relationships[dep_tup]:
                    rel_score = prep_to_relationships[dep_tup][rel]
                    relationship = EventRelationship.create(event_1, event_2, rel, rel_score)
                    relationships.append(relationship)
        return relationships
