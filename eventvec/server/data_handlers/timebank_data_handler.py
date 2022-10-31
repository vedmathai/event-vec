from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_document import TimebankDocument  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_timex import TimebankTimex  # noqa

from eventvec.server.data_readers.timebank_reader.timebank_reader import TimeMLDataReader  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_event import TimebankEvent  # noqa

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
}

rel2rel_simpler = {
    'IS_INCLUDED': 'during',
    'SIMULTANEOUS': 'during',
    'BEFORE': 'before',
    'IDENTITY': 'during',
    'DURING': 'during',
    'ENDED_BY': 'before',
    'BEGUN_BY': 'after',
    'IAFTER': 'before',
    'AFTER': 'after',
    'IBEFORE': 'after',
    'ENDS': 'after',
    'INCLUDES': 'during',
    'DURING_INV': 'during',
    'BEGINS': 'before',
}


class TimeBankBertDataHandler:

    def get_data(self):
        tmdr = TimeMLDataReader()
        timebank_documents = tmdr.timebank_documents()
        data = []
        for document in timebank_documents:
            for tlink in document.tlinks():
                if 't' in tlink.tlink_type():
                    continue
                tlink_type = tlink.tlink_type()
                if tlink_type[0] == 'e':
                    from_event = tlink.event_instance_id()
                    from_sentence = self.event_instance_id2sentence(
                        document, from_event
                    )
                if tlink_type[1:] == '2e':
                    to_event = tlink.related_to_event_instance()
                    to_sentence = self.event_instance_id2sentence(
                        document, to_event
                    )
                if tlink_type[0] == 't':
                    from_time = tlink.time_id()
                    from_sentence = self.time_id2sentence(document, from_time)
                if tlink_type[1:] == '2t':
                    to_time = tlink.related_to_time()
                    to_sentence = self.time_id2sentence(document, to_time)
                data.append({
                    'from_sentence': from_sentence,
                    'to_sentence': to_sentence,
                    'relationship': rel2rel_simpler[tlink.rel_type()],
                })
        return data

    def event_instance_id2sentence(self, document, eiid):
        make_instance = document.event_instance_id2make_instance(eiid)
        eid = make_instance.event_id()
        sseq = []
        if document.is_eid(eid) is True:
            from_sentence = document.eid2sentence(eid)
            for s in from_sentence.sequence():
                if isinstance(s, TimebankEvent):
                    if s.eid() == eid:
                        sseq.extend(['[ENTITY_1]', s.text(), '[/ENTITY_1]'])
                else:
                    sseq.append(s.text())
        return sseq

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


if __name__ == '__main__':
    bdh = TimeBankBertDataHandler()
    bdh.get_data()
