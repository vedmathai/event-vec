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
            items_axis_1 = {i.id(): k for k, i in enumerate(nodes) if i.pos() == pos[0]}
            items_axis_2 = {i.id(): k for k, i in enumerate(nodes) if i.pos() == pos[1]}
            num_nodes1 = len(items_axis_1)
            num_nodes2 = len(items_axis_2)
            matrix_weighted = np.zeros((num_nodes1, num_nodes2))
            matrix = np.zeros((num_nodes1, num_nodes2))
            for node in nodes:
                if node.id() in items_axis_1:
                    for relationship in node.relationships().values():
                        if relationship.relationship_type() != 'NONE':
                            other_node = relationship.other_node(node)
                            if other_node.pos() == pos[1] and other_node.id() in items_axis_2:
                                node_idx = items_axis_1[node.id()]
                                other_node_idx = items_axis_2[other_node.id()]
                                matrix_weighted[node_idx, other_node_idx] = np.log(relationship.count())
                                matrix[node_idx, other_node_idx] = 1

            filename = os.path.join(HEATMAP_FOLDER, 'axis_{}.txt'.format('-'.join(pos).lower()))
            with open(filename, 'wt') as f:
                for item in sorted(items_axis_1.items(), key=lambda x: x[1]):
                    f.write('{} & {} \\\\\n'.format(item[1], item[0]))

            plt.imshow(matrix, cmap='hot', interpolation='nearest')
            filename = os.path.join(HEATMAP_FOLDER, 'heatmap_unweighted_{}_{}_{}.png'.format(notes, '-'.join(pos).lower(), limit))
            plt.savefig(filename)
            plt.imshow(matrix_weighted, cmap='hot', interpolation='nearest')
            filename = os.path.join(HEATMAP_FOLDER, 'heatmap_weighted_{}_{}_{}.png'.format(notes, '-'.join(pos).lower(), limit))
            plt.savefig(filename)
