class ConnectedSubGraph:
    def __init__(self):
        self._nodes = set()

    def add_node(self, node):
        self._nodes.add(node)

    def nodes(self):
        return self._nodes

    def size(self):
        return len(self._nodes)
