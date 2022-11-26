from eventvec.server.model.word_graph.relationship import Relationship
from eventvec.server.model.word_graph.event_trigger import EventTrigger


class WordGraph:
    def __init__(self):
        self._relationships = {}
        self._nodes = {}

    def add_node(self, node: EventTrigger):
        self._nodes[node.id()] = node

    def add_relationship(self, relationship: Relationship):
        self._relationships[relationship.id()] = relationship

    def nodes(self):
        return list(self._nodes.values())

    def relationships(self):
        return list(self._relationships.values())

    def node(self, node_id):
        return self._nodes[node_id]

    def relationship(self, relationship_id):
        return self._relationships[relationship_id]

    def reset(self):
        for node in self._nodes.values():
            node.reset()

    def nodes2relationship(self, node_id_1, node_id_2):
        ids = sorted([node_id_1, node_id_2])
        relationship_id = '{}|{}'.format(*ids)
        return self._relationships[relationship_id]

    def is_node(self, node_id):
        return node_id in self._nodes

    def is_relationship(self, node_id_1, node_id_2):
        ids = sorted([node_id_1, node_id_2])
        relationship_id = '{}|{}'.format(*ids)
        return relationship_id in self._relationships
