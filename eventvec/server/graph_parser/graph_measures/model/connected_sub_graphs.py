class ConnectedSubGraphs:
    def __init__(self):
        self._connected_sub_graphs = []

    def add_connected_sub_graph(self, connected_sub_graph):
        self._connected_sub_graphs.append(connected_sub_graph)

    def connected_sub_graphs(self):
        return self._connected_sub_graphs

    def length(self):
        return len(self._connected_sub_graphs)
