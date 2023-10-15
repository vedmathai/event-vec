import numpy as np
from collections import defaultdict
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_document import TimebankDocument  # noqa
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_timex import TimebankTimex  # noqa
from eventvec.server.data.timebank.timebank_reader.timebank_reader import TimeMLDataReader  # noqa
from eventvec.server.data.timebank.timebank_reader.timebank_dense_reader import TimeBankDenseDataReader  # noqa
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_event import TimebankEvent  # noqa
from eventvec.server.tasks.relationship_classification.datahandlers.model_input.model_input_data import ModelInputData  # noqa
from eventvec.server.tasks.relationship_classification.datahandlers.model_input.model_input_datum import ModelInputDatum  # noqa
from eventvec.server.data.timebank.timebank_reader.te3_gold_reader import TE3GoldDatareader  # noqa
from eventvec.server.data.timebank.timebank_reader.te3_silver_reader import TE3SilverDatareader  # noqa
from eventvec.server.data.timebank.timebank_reader.te3_platinum_reader import TE3PlatinumDatareader  # noqa
from eventvec.server.data.matres.matres_readers.matres_reader import MatresDataReader  # noqa
from eventvec.server.data.timebank.timebank_reader.aquaint_reader import AquaintDatareader  # noqa


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
    'is_included': 'during',
    'simultaneous': 'during',
    'before': 'before',
    'identity': 'during',
    'during': 'during',
    'ended_by': 'during',
    'begun_by': 'during',
    'iafter': 'after',
    'after': 'after',
    'ibefore': 'before',
    'ends': 'during',
    'includes': 'during',
    'during_inv': 'during',
    'begins': 'during',
    'none': 'none',
    'modal': 'modal',
    'evidential': 'evidential',
    'factual': 'factual',
    'equal': 'during',
    "vague": "vague",
    "counter_factive": "counter_factive",
    'factive': 'factive',
    'neg_evidential': 'neg_evidential',
    'conditional': 'conditional',
    None: 'none',
}

rel2rel_opposite = {
    'during': 'during',
    'after': 'before',
    'before': 'after',
    'none': 'none',
    'modal': 'modal',
    'evidential': 'evidential',
    'factual': 'factual',
    'equal': 'during',
    "vague": "vague",
    "counter_factive": "counter_factive",
    'factive': 'factive',
    'neg_evidential': 'neg_evidential',
    'conditional': 'conditional',
    None: 'none',
}

datareaders = {
    'timebank': TimeMLDataReader,
    'timebank_dense': TimeBankDenseDataReader,
    'te3_gold': TE3GoldDatareader,
    'te3_silver': TE3SilverDatareader,
    'te3_platinum': TE3PlatinumDatareader,
}


class TimeBankBertDataHandler:

    def __init__(self):
        self._model_input_data = ModelInputData()
        self._data = []
        self._train_data = []
        self._test_data = []
        self._matres_reader = MatresDataReader()
        self._aquaint_reader = AquaintDatareader()
        self._timebank_reader = TE3GoldDatareader()
        self._timebank_platinum_reader = TE3PlatinumDatareader()

    def load(self):
        self.allocate_train_test_data()

    def load_data(self, documents, train_test):
        data = []
        pos2rel = defaultdict(lambda: defaultdict(int))
        self._matres_dict = self._matres_reader.matres_dict('timebank')
        self._matres_dict.update(self._matres_reader.matres_dict('aquaint'))
        self._matres_dict.update(self._matres_reader.matres_dict('platinum'))
        t0_dict = {}
        for document in documents:
            for tlink in document.tlinks():
                if 't' in tlink.tlink_type():
                    continue
                tlink_type = tlink.tlink_type()
                if tlink_type[0] == 'e':
                    from_event = tlink.event_instance_id()

                if tlink_type[1:] == '2e':
                    to_event = tlink.related_to_event_instance()
                rel_type = tlink.rel_type()
                data = self._process_link(document, from_event, to_event, rel_type, data, t0_dict)

            for slink in document.slinks():
                from_event = slink.event_instance_id()
                to_event = slink.subordinated_event_instance()
                rel_type = slink.rel_type()
                data = self._process_link(document, from_event, to_event, rel_type, data, t0_dict)
        return data

    def _process_link(self, document, from_event, to_event, rel_type, data, t0_dict):

        matres_key = (document.file_name(), from_event.strip('ei'), to_event.strip('ei'))
        matres_key_opp = (document.file_name(), to_event.strip('ei'), from_event.strip('ei'))

        if rel_type in ['EVIDENTIAL', 'MODAL', 'FACTUAL']:
            if matres_key in self._matres_dict:
                rel_type = self._matres_dict[matres_key][-1]
            elif matres_key_opp in self._matres_dict:
                rel_type = self._matres_dict[matres_key_opp][-1]

        rel_type = rel2rel_simpler[rel_type.lower()]
        from_sentence, from_sseq, from_sentence_i, from_token_i, from_start_token_i, from_end_token_i, from_token = self.event_instance_id2sentence(
            document, from_event, 'from',
        )

        to_sentence, to_sseq, to_sentence_i, to_token_i, to_start_token_i, to_end_token_i, to_token = self.event_instance_id2sentence(
            document, to_event, 'to',
        )


        token_order = self.token_order(from_sentence_i, from_token_i, to_sentence_i, to_token_i)

        if len(to_sentence.text()) == 0 or len(from_sentence.text()) == 0:
            return data
        
        if token_order is False:
            from_sentence, to_sentence = to_sentence, from_sentence
            from_sseq, to_sseq = to_sseq, from_sseq
            rel_type = rel2rel_opposite[rel_type]
            from_start_token_i, to_start_token_i = to_start_token_i, from_start_token_i
            from_end_token_i, to_end_token_i = to_end_token_i, from_end_token_i

        model_input_datum = ModelInputDatum()
        model_input_datum.set_from_original_sentence(from_sentence.text())
        model_input_datum.set_to_original_sentence(to_sentence.text())
        model_input_datum.set_from_sentence(from_sseq)
        model_input_datum.set_to_sentence(to_sseq)
        model_input_datum.set_relationship(rel_type)
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
        return data

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
                        token = s
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
                        sseq.extend(['entity2', s.text(), 'entity1'])
                else:
                    sseq.append(s.text())
        return sseq

    def token_order(self, from_sentence_i, from_token_i, to_sentence_i,
                    to_token_i):
        if from_sentence_i < to_sentence_i:
            return True
        elif from_sentence_i > to_sentence_i:
            return False
        elif from_sentence_i == to_sentence_i:
            if from_token_i < to_token_i:
                return True
            else:
                return False

    def allocate_train_test_data(self):
        dr = datareaders.get(DATANAME)
        dr = dr()
        train_documents = self._timebank_reader.timebank_documents() + self._aquaint_reader.timebank_documents()
        test_documents = self._timebank_platinum_reader.timebank_documents()
        train_data = self.load_data(train_documents, 'train')
        self._model_input_data.set_train_data(train_data)
        test_data = self.load_data(test_documents, 'test')
        self._model_input_data.set_test_data(test_data)

    def model_input_data(self):
        return self._model_input_data
