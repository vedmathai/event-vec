from eventvec.server.model.event_models.event_model import Event
from eventvec.server.train.vectorizer.dep_parser_model import follow_down
from eventvec.server.model.event_models.path_libary import subject_paths, object_paths, verb_paths


class EventExtractor():
    def extract(self, verb_node, psentence):
        subject_nodes = follow_down(verb_node, subject_paths)
        object_nodes = follow_down(verb_node, object_paths)
        verb_nodes = follow_down(verb_node, verb_paths)
        event = Event.create_from_paths(
            subject_nodes=subject_nodes,
            object_nodes=object_nodes,
            verb_nodes=verb_nodes,
        )
        return event
