from eventvec.server.model.word_graph.word_graph import WordGraph
from eventvec.server.model.word_graph.relationship import Relationship
from eventvec.server.model.word_graph.event_trigger import EventTrigger
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_event import TimebankEvent  # noqa


class EventGraphCreator():
    def __init__(self):
        self._graph = WordGraph()

    def create_graph(self, documents):
        for document in documents:
            for tlink in document.tlinks():
                tlink_type = tlink.tlink_type()
                relationship = tlink.rel_type()
                if tlink_type == 'e2e' and relationship != 'none':
                    from_event = tlink.event_instance_id()
                    from_make_instance = document.event_instance_id2make_instance(from_event)
                    from_pos = from_make_instance.pos()

                    from_token, from_sentence_i, from_token_i, from_eid = self.event_instance_id2sentence(
                        document, from_event,
                    )
                    to_event = tlink.related_to_event_instance()
                    to_make_instance = document.event_instance_id2make_instance(to_event)
                    to_pos = to_make_instance.pos()
                    to_token, to_sentence_i, to_token_i, to_eid = self.event_instance_id2sentence(
                        document, to_event,
                    )

                    token_order = self.token_order(
                        from_sentence_i, from_token_i, to_sentence_i,
                        from_sentence_i
                    )
                    relationship = tlink.rel_type()
                    if relationship == 'none':
                        continue
                    if self._graph.is_node(from_token) is False:
                        from_node = EventTrigger()
                        from_node.set_id(from_event)
                        from_node.set_word(from_token)
                        from_node.set_pos(from_pos)
                        self._graph.add_node(from_node)
                    from_node = self._graph.node(from_token)
                    from_node.increment_count()
                    from_node.add_source_document(document.file_name())

                    if self._graph.is_node(to_token) is False:
                        to_node = EventTrigger()
                        to_node.set_id(to_event)
                        to_node.set_word(to_token)
                        to_node.set_pos(to_pos)
                        self._graph.add_node(to_node)
                    to_node = self._graph.node(to_token)
                    to_node.increment_count()
                    to_node.add_source_document(document.file_name())

                    relationship_obj = Relationship()
                    relationship_obj.set_node1(from_node)
                    relationship_obj.set_node2(to_node)
                    relationship_obj.set_relationship_type(relationship)
                    from_node.add_relationship(relationship_obj)
                    to_node.add_relationship(relationship_obj)
                    relationship_obj.increment_count()
                    self._graph.add_relationship(relationship_obj)
        return self._graph

    def event_instance_id2sentence(self, document, eiid):
        make_instance = document.event_instance_id2make_instance(eiid)
        eid = make_instance.event_id()
        word = None
        from_sentence = None
        token_i = None
        sentence_i = None
        if document.is_eid(eid) is True:
            from_sentence = document.eid2sentence(eid)
            sentence_i = from_sentence.sentence_i()
            for s in from_sentence.sequence():
                if isinstance(s, TimebankEvent):
                    if s.eid() == eid:
                        token_i = s.sentence_token_i()
                        word = s.text().lower()
        return word, sentence_i, token_i, eid

    def token_order(self, from_sentence_i, from_token_i, to_sentence_i,
                    to_token_i):
        if from_sentence_i < to_sentence_i:
            return 0
        elif from_sentence_i > to_sentence_i:
            return 1
        else:
            if from_token_i < to_token_i:
                return 0
            else:
                return 1
