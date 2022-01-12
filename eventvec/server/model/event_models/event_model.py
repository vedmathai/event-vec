class Event:
    def __init__(self):
        self._subject_nodes = None
        self._object_nodes = None
        self._verb_nodes = None
        self._root_node = None

    def to_dict(self):
        return {
            "subject_nodes": [i.orth() for i in sorted(self._subject_nodes, key=key_fn)],
            "object_nodes": [i.orth() for i in sorted(self._object_nodes, key=key_fn)],
            "verb_nodes": [i.orth() for i in sorted(self._verb_nodes, key=key_fn)],
            "root_node": self._root_node.orth(),
        }

    def __repr__(self) -> str:
        return str(self.to_dict())

    def root_node(self):
        return self._root_node

    @staticmethod
    def create_from_paths(subject_nodes, object_nodes, verb_nodes, root_node):
        event = Event()
        event._subject_nodes = subject_nodes
        event._object_nodes = object_nodes
        event._verb_nodes = verb_nodes
        event._root_node = root_node
        return event

def key_fn(i):
    return i.i()