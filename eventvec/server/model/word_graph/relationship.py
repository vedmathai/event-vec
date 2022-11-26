class Relationship():
    def __init__(self):
        self._node1 = None
        self._node2 = None
        self._relationship_type = None
        self._count = 0

    def set_node1(self, node1):
        self._node1 = node1

    def set_node2(self, node2):
        self._node2 = node2

    def set_relationship_type(self, relationship_type):
        self._relationship_type = relationship_type

    def node1(self):
        return self._node1

    def node2(self):
        return self._node2

    def count(self):
        return self._count

    def relationship_type(self):
        return self._relationship_type

    def id(self):
        ids = sorted([self.node1().id(), self.node2().id()])
        return '{}|{}'.format(*ids)

    def increment_count(self):
        self._count += 1

    def other_node(self, node):
        if node == self._node1:
            return self._node2
        if node == self._node2:
            return self._node1
