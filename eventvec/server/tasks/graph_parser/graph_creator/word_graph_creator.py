import os
import re
from tqdm import tqdm

from eventvec.server.model.word_graph.word_graph import WordGraph
from eventvec.server.model.word_graph.relationship import Relationship
from eventvec.server.model.word_graph.event_trigger import EventTrigger
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_event import TimebankEvent  # noqa

WINDOW_SIZE = 512
WORD_LISTS_FOLDER = 'eventvec/server/graph_parser/word_lists'


class WordGraphCreator():
    def __init__(self):
        self._graph = WordGraph()

    def word_lists(self):
        noun_file = os.path.join(WORD_LISTS_FOLDER, 'all_nouns.txt')
        nouns = set()
        with open(noun_file, 'rt') as f:
            for line in f:
                nouns.add(line.strip())
        verb_file = os.path.join(WORD_LISTS_FOLDER, 'all_verbs.txt')
        verbs = set()
        with open(verb_file, 'rt') as f:
            for line in f:
                verbs.add(line.strip())
        stopwords_file = os.path.join(WORD_LISTS_FOLDER, 'stopwords.txt')
        stopwords = set()
        with open(stopwords_file, 'rt') as f:
            for line in f:
                stopwords.add(line.strip())
        return nouns, verbs, stopwords

    def create_graph(self, documents_location):
        nouns, verbs, stopwords = self.word_lists()
        nouns = nouns - stopwords
        verbs = verbs - stopwords
        together = nouns | verbs
        documents_list = os.listdir(documents_location)
        for document in tqdm(documents_list[0:10]):
            document_location = os.path.join(documents_location, document)
            with open(document_location) as f:
                text = f.read()
                split_text = text.lower().split()
                total_words = len(split_text)
                window_dict = {}
                right = 0
                for right in range(total_words):
                    left = max(-1, right - WINDOW_SIZE)
                    word = split_text[right]
                    word = re.sub(r'[^a-z]', '', word)
                    if word in together:
                        if word in nouns:
                            pos = 'NOUN'
                        if word in verbs:
                            pos = 'VERB'
                        if self._graph.is_node(word) is False:
                            from_node = EventTrigger()
                            from_node.set_id(word)
                            from_node.set_word(word)
                            from_node.set_pos(pos)
                            self._graph.add_node(from_node)
                        from_node = self._graph.node(word)
                        from_node.increment_count()
                        from_node.add_source_document(document)

                        for window_word in window_dict:
                            if window_word in nouns:
                                pos = 'NOUN'
                            if window_word in verbs:
                                pos = 'VERB'
                            if self._graph.is_node(window_word) is False:
                                to_node = EventTrigger()
                                to_node.set_id(window_word)
                                to_node.set_word(window_word)
                                to_node.set_pos(pos)
                                self._graph.add_node(to_node)
                            to_node = self._graph.node(window_word)
                            to_node.add_source_document(document)
                            is_relationship = self._graph.is_relationship(from_node.id(), to_node.id())
                            if is_relationship is False:
                                relationship_obj = Relationship()
                                relationship_obj.set_node1(from_node)
                                relationship_obj.set_node2(to_node)
                                from_node.add_relationship(relationship_obj)
                                to_node.add_relationship(relationship_obj)
                                self._graph.add_relationship(relationship_obj)
                            relationship = self._graph.nodes2relationship(from_node.id(), to_node.id())
                            relationship.increment_count()

                    if left >= 0:
                        left_word = split_text[left]
                        left_word = re.sub(r'[^a-z]', '', left_word)
                        if left_word in window_dict:
                            if window_dict[left_word] == 1:
                                del window_dict[left_word]
                            else:
                                window_dict[left_word] -= 1

                    if word in together:
                        if word not in window_dict:
                            window_dict[word] = 0
                        window_dict[word] += 1

        return self._graph
