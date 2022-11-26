import numpy as np

from eventvec.server.graph_parser.graph_measures.model.connected_sub_graph import ConnectedSubGraph
from eventvec.server.graph_parser.graph_measures.model.connected_sub_graphs import ConnectedSubGraphs


class ClusterMeasures:
    def __init__(self):
        self._connected_sub_graphs = ConnectedSubGraphs()

    def measure(self, graph):
        for pos in ['NOUN', 'VERB']:
            print(' ' * 4, pos)
            connected_sub_graphs = self.create_sub_graphs(graph, pos)
            sub_graph_ratio, weighted_sub_graph_ratio = self.measure_sub_graphs(connected_sub_graphs)
            print(' ' * 8, 'Sub Graph Ratio', sub_graph_ratio)
            print(' ' * 8, 'Weighted Sub Graph Ratio', weighted_sub_graph_ratio)
            relationship_ratio, relationship_ratio_weighted = self.relationship_ratio(graph, pos)
            print(' ' * 8, 'Relationship Ratio', relationship_ratio)
            print(' ' * 8, 'Relationship Ratio Weighted', relationship_ratio_weighted)


    def measure_sub_graphs(self, connected_sub_graphs):
        total_nodes = set()
        for connected_sub_graph in connected_sub_graphs.connected_sub_graphs():
            total_nodes |= connected_sub_graph.nodes()
        total_graphs = connected_sub_graphs.length()
        total_nodes_weighted = 0
        total_graphs_weighted = 0
        for sub_graph in connected_sub_graphs.connected_sub_graphs():
            max_count = 0
            for node in sub_graph.nodes():
                max_count = max(node.count(), max_count)
                total_nodes_weighted += node.count()
            total_graphs_weighted += max_count
        graph_ratio = 1 - (float(total_graphs) / len(total_nodes))
        weighted_graph_ratio = 1 - (float(total_graphs_weighted) / total_nodes_weighted)
        return graph_ratio, weighted_graph_ratio

    def create_sub_graphs(self, graph, pos):
        connected_sub_graphs = ConnectedSubGraphs()
        for node in graph.nodes():
            connected_sub_graph = ConnectedSubGraph()
            source_documents = set()
            if node.visited() is True or node.pos() != pos:
                continue
            to_visit = [node]
            connected_sub_graph.add_node(node)
            while len(to_visit) > 0:
                curr = to_visit.pop()
                source_documents |= curr.source_documents()
                if curr.visited() is True:
                    continue
                curr.set_visited_true()
                for relationship_id in curr.relationships():
                    relationship = curr.relationship(relationship_id)
                    relationship_type = relationship.relationship_type()
                    other_node = relationship.other_node(curr)
                    if other_node is not None and relationship_type != 'NONE' and other_node.pos() == pos:
                        connected_sub_graph.add_node(other_node)
                        to_visit.append(other_node)
            connected_sub_graphs.add_connected_sub_graph(connected_sub_graph)
        graph.reset()
        return connected_sub_graphs

    def relationship_ratio(self, graph, pos):
        interested_relationships = 0
        interested_relationships_weighted = 0
        all_nodes = sorted(graph.nodes(), key=lambda x: x.id())
        total_relationships_weighted = 0
        for nodei in range(len(all_nodes)):
            for nodej in range(nodei, len(all_nodes)):
                inode = all_nodes[nodei]
                jnode = all_nodes[nodej]
                total_relationships_weighted += inode.count() * jnode.count()
        total_relationships = len(all_nodes) * (len(all_nodes) - 1)
        for relationship in graph.relationships():
            if relationship.relationship_type() != 'NONE':
                node1 = relationship.node1()
                node2 = relationship.node2()
                if node1.pos() == pos and node2.pos() == pos:
                    interested_relationships_weighted += relationship.count()
                    interested_relationships += 1
        relationship_ratio = float(interested_relationships) / total_relationships
        relationship_ratio_weighted = float(interested_relationships_weighted) / total_relationships_weighted
        return relationship_ratio, relationship_ratio_weighted
