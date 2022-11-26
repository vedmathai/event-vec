import matplotlib.pyplot as plt
import numpy as np
import os

from eventvec.server.graph_parser.graph_measures.cluster_measures import ClusterMeasures  # noqa

HEATMAP_FOLDER = 'eventvec/server/graph_parser/heatmaps'


class GraphClusterHeatmap:

    def create_heatmap(self, graph, pos, notes):
        cluster_measures = ClusterMeasures()
        connected_sub_graphs = cluster_measures.create_sub_graphs(graph, pos)
        connected_sub_graphs = sorted(connected_sub_graphs.connected_sub_graphs(), key=lambda x: len(x.nodes()), reverse=True)
        nodes = []
        for csg in connected_sub_graphs:
            sorted_nodes = sorted(csg.nodes(), key=lambda x: len(x.relationships()), reverse=True)
            nodes.extend(sorted_nodes)
        for limit in [150, 500]:
            items = {i.id(): k for k, i in enumerate(nodes)}
            print(items)
            num_nodes = len(items)
            matrix_weighted = np.zeros((num_nodes, num_nodes))
            matrix = np.zeros((num_nodes, num_nodes))
            for node in nodes:
                if node.id() in items:
                    for relationship in node.relationships().values():
                        if relationship.relationship_type() != 'NONE':
                            other_node = relationship.other_node(node)
                            if other_node.pos() == pos and other_node.id() in items:
                                node_idx = items[node.id()]
                                other_node_idx = items[other_node.id()]
                                matrix_weighted[node_idx, other_node_idx] = np.log(relationship.count())
                                matrix[node_idx, other_node_idx] = 1

            filename = os.path.join(HEATMAP_FOLDER, 'axis_{}.txt'.format(pos.lower()))
            with open(filename, 'wt') as f:
                for item in sorted(items.items(), key=lambda x: x[1]):
                    f.write('{} & {} \\\\\n'.format(item[1], item[0]))

            plt.imshow(matrix, cmap='hot', interpolation='nearest')
            filename = os.path.join(HEATMAP_FOLDER, 'heatmap_unweighted_{}_{}_{}.png'.format(notes, pos.lower(), limit))
            plt.savefig(filename)
            plt.imshow(matrix_weighted, cmap='hot', interpolation='nearest')
            filename = os.path.join(HEATMAP_FOLDER, 'heatmap_weighted_{}_{}_{}.png'.format(notes, pos.lower(), limit))
            plt.savefig(filename)
