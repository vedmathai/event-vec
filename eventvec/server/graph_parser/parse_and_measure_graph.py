from collections import defaultdict
import os

from eventvec.server.data_readers.timebank_reader.timebank_reader import TimeMLDataReader  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_dense_reader import TimeBankDenseDataReader  # noqa
from eventvec.server.graph_parser.graph_creator.event_graph_creator import EventGraphCreator  # noqa
from eventvec.server.graph_parser.graph_creator.word_graph_creator import WordGraphCreator  # noqa
from eventvec.server.graph_parser.graph_measures.cluster_measures import ClusterMeasures  # noqa
from eventvec.server.graph_parser.graph_clusterer.graph_cluster_heatmap import GraphClusterHeatmap  # noqa

datareaders = {
    'timebank': TimeMLDataReader,
    'timebank_dense': TimeBankDenseDataReader,
}

documents_folder = '/home/lalady6977/oerc/projects/data/books1/epubtxt'


def parse_and_measure_graph():
    datareader = datareaders['timebank_dense']
    datareader = datareader()
    train_documents = datareader.train_documents()
    print('Event Cluster')
    egc = EventGraphCreator()
    event_graph = egc.create_graph(train_documents)
    all_nouns, all_verbs = create_interested_words(event_graph)
    gch = GraphClusterHeatmap()
    gch.create_heatmap(event_graph, 'NOUN', 'event_graph')
    gch.create_heatmap(event_graph, 'VERB', 'event_graph')
    cluster_measures = ClusterMeasures()
    cluster_measures.measure(event_graph)
    print('Word Cluster')
    wgc = WordGraphCreator()
    word_graph = wgc.create_graph(documents_folder)
    cluster_measures = ClusterMeasures()
    cluster_measures.measure(word_graph)
    gch.create_heatmap(word_graph, 'VERB', 'word_graph')
    gch.create_heatmap(word_graph, 'NOUN', 'word_graph')

    # create the heatmap
    # create the connectedness measure
    # create the cluserting measure
    # repeat for verbs
    # repeat for larger document


def create_interested_words(graph):
    all_nouns = set()
    all_verbs = set()
    word_lists_folder = 'eventvec/server/graph_parser/word_lists'
    for node in graph.nodes():
        if node.pos() == 'NOUN':
            all_nouns.add(node.word().lower())
        if node.pos() == 'VERB':
            all_verbs.add(node.word().lower())
    all_nouns_file = os.path.join(word_lists_folder, 'all_nouns.txt')
    all_verbs_file = os.path.join(word_lists_folder, 'all_verbs.txt')
    with open(all_nouns_file, 'wt') as f:
        for word in sorted(list(all_nouns)):
            f.write(word + '\n')
    with open(all_verbs_file, 'wt') as f:
        for word in sorted(list(all_verbs)):
            f.write(word + '\n')
    return all_nouns, all_verbs


if __name__ == '__main__':
    parse_and_measure_graph()
