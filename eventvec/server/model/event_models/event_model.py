class Event:
    def __init__(self):
        self._subject_nodes = None
        self._object_nodes = None
        self._verb_nodes = None
        self._root_node = None
        self._verb_tensor = None
        self._object_tensor = None
        self._subject_tensor = None

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

    def verb_nodes(self):
        return self._verb_nodes

    def object_nodes(self):
        return self._object_nodes

    def subject_nodes(self):
        return self._subject_nodes

    def set_object_tensor(self, object_tensor):
        self._object_tensor = object_tensor

    def set_verb_tensor(self, verb_tensor):
        self._verb_tensor = verb_tensor

    def set_subject_tensor(self, subject_tensor):
        self._subject_tensor = subject_tensor

    def subject_tensor(self):
        return self._subject_tensor

    def verb_tensor(self):
        return self._verb_tensor

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