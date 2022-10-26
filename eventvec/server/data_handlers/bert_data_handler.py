from collections import defaultdict
from xml.dom.minidom import Document
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_timex import TimebankTimex

from eventvec.server.data_readers.timebank_reader.timebank_reader import TimeMLDataReader  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_event import TimebankEvent  # noqa


class BertDataHandler:
    def get_data(self):
        return self.get_timebank_data()

    def get_timebank_data(self):
        tmdr = TimeMLDataReader()
        timebank_documents = tmdr.timebank_documents()
        for document in timebank_documents:
            for tlink in document.tlinks():
                if 't' not in tlink.tlink_type():
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

                print(' '.join(from_sentence), tlink.rel_type(), ' '.join(to_sentence), '\n')

    def event_instance_id2sentence(self, document, eiid):
        make_instance = document.event_instance_id2make_instance(eiid)
        eid = make_instance.event_id()
        sseq = []
        if document.is_eid(eid) is True:
            from_sentence = document.eid2sentence(eid)
            for s in from_sentence.sequence():
                if isinstance(s, TimebankEvent):
                    if s.eid() == eid:
                        sseq.extend(['[e]', s.text(), '[/e]'])
                else:
                    sseq.append(s.text())
        return sseq

    def time_id2sentence(self, document: Document, time_id):
        sseq = []
        if document.is_time_id(time_id) is True:
            from_sentence = document.time_id2sentence(time_id)
            for s in from_sentence.sequence():
                if isinstance(s, TimebankTimex):
                    if s.tid() == time_id:
                        sseq.extend(['[e]', s.text(), '[/e]'])
                else:
                    sseq.append(s.text())
        return sseq


if __name__ == '__main__':
    bdh = BertDataHandler()
    bdh.get_data()
