import numpy as np
from collections import defaultdict
import pprint

from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_document import TimebankDocument  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_timex import TimebankTimex  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_reader import TimeMLDataReader  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_dense_reader import TimeBankDenseDataReader  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_event import TimebankEvent  # noqa
from eventvec.server.data_handlers.model_input.model_input_data import ModelInputData  # noqa
from eventvec.server.data_handlers.model_input.model_input_datum import ModelInputDatum  # noqa

DATANAME = "timebank_dense"


rel2rel = {
    'IS_INCLUDED': 'is_included',
    'SIMULTANEOUS': 'simultaneous',
    'BEFORE': 'before',
    'IDENTITY': 'identity',
    'DURING': 'during',
    'ENDED_BY': 'ended_by',
    'BEGUN_BY': 'begun_by',
    'IAFTER': 'i_after',
    'AFTER': 'after',
    'IBEFORE': 'i_before',
    'ENDS': 'ends',
    'INCLUDES': 'includes',
    'DURING_INV': 'during_inv',
    'BEGINS': 'begins',
    'NONE': 'none',
}

rel2rel_opposite = {
    'IS_INCLUDED': 'includes',
    'SIMULTANEOUS': 'simultaneous',
    'BEFORE': 'after',
    'AFTER': 'before',
    'INCLUDES': 'is_included',
    'NONE': 'none',
}

rel2rel_simpler = {
    'IS_INCLUDED': 'during',
    'SIMULTANEOUS': 'during',
    'BEFORE': 'before',
    'IDENTITY': 'identity',
    'DURING': 'during',
    'ENDED_BY': 'during',
    'BEGUN_BY': 'during',
    'IAFTER': 'after',
    'AFTER': 'after',
    'IBEFORE': 'before',
    'ENDS': 'during',
    'INCLUDES': 'during',
    'DURING_INV': 'during',
    'BEGINS': 'during',
    'NONE': 'none',
}

datareaders = {
    'timebank': TimeMLDataReader,
    'timebank_dense': TimeBankDenseDataReader,
}


class TimeBankBertDataHandler:

    def __init__(self):
        self._model_input_data = ModelInputData()
        self._data = []
        self._train_data = []
        self._test_data = []

    def load(self):
        self.allocate_train_test_data()

    def load_data(self, documents, train_test):
        data = []
        rels = set()
        nouns = set()
        pos2rel = defaultdict(lambda: defaultdict(int))
        for document in documents:
            for tlink in document.tlinks():
                if 't' in tlink.tlink_type():
                    continue
                tlink_type = tlink.tlink_type()
                if tlink_type[0] == 'e':
                    from_event = tlink.event_instance_id()
                    make_instance = document.event_instance_id2make_instance(from_event)
                    first_pos = make_instance.pos()
                    if (train_test == 'train' and make_instance.pos() not in ['NOUN']) or train_test == 'test' and make_instance.pos() not in ['NOUN']:
                        pass

                    from_original_sentence, from_sentence, from_sentence_i, from_token_i, from_start_token_i, from_end_token_i, from_token = self.event_instance_id2sentence(
                        document, from_event, 'from',
                    )
                if tlink_type[1:] == '2e':
                    to_event = tlink.related_to_event_instance()
                    make_instance = document.event_instance_id2make_instance(to_event)
                    second_pos = make_instance.pos()
                    if (train_test == 'train' and make_instance.pos() not in ['VERB']) or train_test == 'test' and make_instance.pos() not in ['VERB']:
                        pass
                    to_original_sentence, to_sentence, to_sentence_i, to_token_i, to_start_token_i, to_end_token_i, to_token = self.event_instance_id2sentence(
                        document, to_event, 'to',
                    )
                if tlink_type[0] == 't':
                    from_time = tlink.time_id()
                    from_sentence = self.time_id2sentence(document, from_time)
                if tlink_type[1:] == '2t':
                    to_time = tlink.related_to_time()
                    to_sentence = self.time_id2sentence(document, to_time)
                rels.add(tlink.rel_type())
                if len(to_sentence) == 0 or len(from_sentence) == 0:
                    continue

                token_order = self.token_order(
                    from_sentence_i, from_token_i, to_sentence_i,
                    from_sentence_i
                )
                if to_sentence_i != from_sentence_i:
                    pass
                relationship = rel2rel[tlink.rel_type()]
                if relationship == 'none':
                    continue


                if token_order == 1:
                    from_original_sentence, to_original_sentence = to_original_sentence, from_original_sentence
                    from_sentence, to_sentence = to_sentence, from_sentence
                    relationship = rel2rel_opposite[tlink.rel_type()]
                    from_start_token_i, to_start_token_i = to_start_token_i, from_start_token_i
                    from_end_token_i, to_end_token_i = to_end_token_i, from_end_token_i
                    token_order = 0
                    pos2rel[(second_pos, first_pos)][relationship] += 1
                else:
                    pos2rel[(first_pos, second_pos)][relationship] += 1


                model_input_datum = ModelInputDatum()
                model_input_datum.set_from_original_sentence(from_original_sentence.text())
                model_input_datum.set_to_original_sentence(to_original_sentence.text())
                model_input_datum.set_from_sentence(from_sentence)
                model_input_datum.set_to_sentence(to_sentence)

                model_input_datum.set_relationship(relationship)
                model_input_datum.set_from_entity_start_token_i(
                    from_start_token_i
                )
                model_input_datum.set_from_entity_end_token_i(
                    from_end_token_i
                )
                model_input_datum.set_to_entity_start_token_i(
                    to_start_token_i
                )
                model_input_datum.set_to_entity_end_token_i(
                    to_end_token_i
                )
                model_input_datum.set_token_order(token_order)
                data.append(model_input_datum)
        print(' ' * 4)
        for pos in pos2rel:
            print(pos, pos2rel[pos])
            print('-' * 32)
            for itemi, item in enumerate(sorted(pos2rel[pos].items(), key=lambda x: x[1])):
                print(pos, itemi, item)
        return data, nouns

    def event_instance_id2sentence(self, document, eiid, event_point):
        make_instance = document.event_instance_id2make_instance(eiid)
        eid = make_instance.event_id()
        sseq = []
        sentence_i = None
        token_i = None
        start_token_i = None
        end_token_i = None
        from_sentence = None
        token = None
        if event_point == 'from':
            tags = ('entity1', 'entity1')
        elif event_point == 'to':
            tags = ('entity2', 'entity2')
        if document.is_eid(eid) is True:
            from_sentence = document.eid2sentence(eid)
            sentence_i = from_sentence.sentence_i()
            for s in from_sentence.sequence():
                if isinstance(s, TimebankEvent):
                    if s.eid() == eid:
                        sseq.extend([tags[0], s.text(), tags[1]])
                        token_i = s.sentence_token_i()
                        start_token_i = s.start_token_i()
                        end_token_i = s.end_token_i()
                        token = s.text()
                    else:
                        sseq.append(s.text())
                else:
                    sseq.append(s.text())
        return from_sentence, sseq, sentence_i, token_i, start_token_i, end_token_i, token

    def time_id2sentence(self, document: TimebankDocument, time_id):
        sseq = []
        if document.is_time_id(time_id) is True:
            from_sentence = document.time_id2sentence(time_id)
            for s in from_sentence.sequence():
                if isinstance(s, TimebankTimex):
                    if s.tid() == time_id:
                        sseq.extend(['[ENTITY_2]', s.text(), '[/ENTITY_2]'])
                else:
                    sseq.append(s.text())
        return sseq

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

    def allocate_train_test_data(self):
        dr = datareaders.get(DATANAME)
        dr = dr()
        train_documents = dr.train_documents()
        test_documents = dr.test_documents()
        train_data, train_nouns = self.load_data(train_documents, 'train')
        self._model_input_data.set_train_data(train_data)
        test_data, test_nouns = self.load_data(test_documents, 'test')
        self._model_input_data.set_test_data(test_data)

    def model_input_data(self):
        return self._model_input_data
