from collections import defaultdict
from eventvec.server.data_handlers.data_handler import PREP_LIST

from eventvec.server.entry_points.vectorizer.dep_parser_model import get_path
from eventvec.server.model.event_models.event_relationship_model import EventRelationship
from eventvec.utils.timebank_prepositions import (
    prep_to_relationships, prep_to_opposite_relationships,
    opposite_dict
)

PREP_LIST = ['AFTER', 'BEFORE', 'IS_INCLUDED'][0:3]

class EventRelationshipExtractor:
    def extract(self, sentence, event_1, event_2):
        relationships = []
        node_1 = event_1.root_node()
        node_2 = event_2.root_node()
        dep = get_path(sentence, node_1.i(), node_2.i())
        dep_tup = []
        for ii, i in enumerate(dep):
            if i.dep() == 'prep':
                dep_tup += [i.lemma()]
        dep_tup = '|'.join(dep_tup)
        if len(dep_tup) > 0 and len(dep) > 0:
            if dep_tup in prep_to_relationships:
                relationship = EventRelationship.create(event_1, event_2)
                for rel in prep_to_relationships[dep_tup]:
                    if True or rel in PREP_LIST:
                        rel_score = prep_to_relationships[dep_tup][rel]
                        relationship.add_relationship_type(rel, rel_score)
                relationship.normalize_distribution()
                relationships.append(relationship)

                opp_relationship = EventRelationship.create(event_2, event_1)
                for rel in prep_to_relationships[dep_tup]:
                    if True or rel in PREP_LIST:
                        opp_rel = opposite_dict[rel]
                        rel_score = prep_to_opposite_relationships[dep_tup][opp_rel]
                        opp_relationship.add_relationship_type(opp_rel, rel_score)
                opp_relationship.normalize_distribution()
                relationships.append(opp_relationship)
            #else:
            #    print('missing', dep_tup)
        return relationships
