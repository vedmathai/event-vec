from bs4 import BeautifulSoup
from typing import List


from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_sentence import TimebankSentence  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_tlink import TimebankTlink  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_slink import TimebankSlink  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_alink import TimebankAlink  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_makeinstance import MakeInstance  # noqa


class TimebankDocument:
    def __init__(self):
        self._file_name = None
        self._tlinks = []
        self._slinks = []
        self._alinks = []
        self._timebank_sentences = []
        self._make_instances = []
        self._eid2event = {}
        self._eid2sentence = {}
        self._time_id2timex3 = {}
        self._time_id2sentence = {}
        self._event_id2make_instance = {}
        self._event_instance_id2make_instance = {}
        self._event_id2tlink = {}
        self._time_id2tlink = {}
        self._event_id2slink = {}
        self._time_id2slink = {}
        self._event_id2alink = {}
        self._time_id2alink = {}

    def file_name(self):
        return self._file_name

    def set_file_name(self, file_name):
        self._file_name = file_name

    def slinks(self):
        return self._slinks

    def add_slink(self, slink):
        self._slinks.append(slink)

    def set_slinks(self, slinks):
        self._slinks = slinks

    def alinks(self):
        return self._alinks

    def add_alink(self, alink):
        self._alinks.append(alink)

    def set_alinks(self, alinks):
        self._alinks = alinks

    def tlinks(self) -> List[TimebankTlink]:
        return self._tlinks

    def add_tlink(self, tlink):
        self._tlinks.append(tlink)

    def set_tlinks(self, tlinks):
        self._tlinks = tlinks

    def timebank_sentences(self):
        return self._timebank_sentences

    def add_timebank_sentence(self, timebank_sentence):
        self._timebank_sentences.append(timebank_sentence)

    def set_timebank_sentences(self, timebank_sentences):
        self._timebank_sentences = timebank_sentences

    def make_instances(self):
        return self._make_instances

    def add_make_instance(self, make_instance):
        self._make_instances.append(make_instance)

    def set_make_instances(self, make_instances):
        self._make_instances = make_instances

    def add_eid2event(self, eid, event):
        self._eid2event[eid] = event

    def add_eid2sentence(self, eid, sentence):
        self._eid2sentence[eid] = sentence

    def add_time_id2timex3(self, time_id, event):
        self._time_id2timex3[time_id] = event

    def add_time_id2sentence(self, time_id, sentence):
        self._time_id2sentence[time_id] = sentence

    def add_event_id2make_instance(self, event_id, make_instance):
        self._event_id2make_instance[event_id] = make_instance

    def add_event_instance_id2make_instance(self, event_instance_id, make_instance):  # noqa
        self._event_instance_id2make_instance[event_instance_id] = make_instance  # noqa

    def add_event_instance_id2tlink(self, event_instance_id, tlink):
        self._event_id2tlink[event_instance_id] = tlink

    def add_time_id2tlink(self, time_id, tlink):
        self._time_id2tlink[time_id] = tlink

    def add_event_instance_id2slink(self, event_instance_id, slink):
        self._event_id2slink[event_instance_id] = slink

    def add_time_id2slink(self, time_id, slink):
        self._time_id2slink[time_id] = slink

    def add_event_instance_id2alink(self, event_instance_id, alink):
        self._event_id2alink[event_instance_id] = alink

    def add_time_id2alink(self, time_id, alink):
        self._time_id2alink[time_id] = alink

    def eid2event(self, eid):
        return self._eid2event[eid]

    def eids(self):
        return list(self._eid2event.keys())

    def is_eid(self, eid):
        return eid in self._eid2event

    def eid2sentence(self, eid):
        return self._eid2sentence.get(eid)

    def event_id2make_instance(self, event_id):
        return self._event_id2make_instance[event_id]

    def event_instance_id2make_instance(self, event_instance_id):
        return self._event_instance_id2make_instance[event_instance_id]

    def event_instance_id2tlink(self, event_instance_id):
        return self._event_id2tlink[event_instance_id]

    def is_time_id(self, time_id):
        return time_id in self._time_id2timex3

    def time_id2sentence(self, time_id):
        return self._time_id2sentence.get(time_id)

    def time_id2timex3(self, time_id):
        return self._time_id2timex3[time_id]

    def time_id2tlink(self, time_id):
        return self._event_id2tlink[time_id]

    def event_instance_id2slink(self, event_instance_id):
        return self._event_id2slink[event_instance_id]

    def time_id2slink(self, time_id):
        return self._time2slink[time_id]

    def event_instance_id2alink(self, event_instance_id):
        return self._event_id2alink[event_instance_id]

    def time_id2alink(self, time_id):
        return self._time_id2alink[time_id]

    @staticmethod
    def from_xml(document):
        timebank_document = TimebankDocument()
        soup = BeautifulSoup(document, 'lxml')

        docno = soup.find('docno')
        timebank_document.set_file_name(docno.text)

        sentences = list(soup.find_all('s'))
        for s in sentences:
            timebank_sentence = TimebankSentence.from_bs_obj(
                s, timebank_document
            )
            timebank_document.add_timebank_sentence(timebank_sentence)

        tlinks = list(soup.find_all('tlink'))
        for tlink in tlinks:
            timebank_tlink = TimebankTlink.from_bs_obj(
                tlink, timebank_document
            )
            timebank_document.add_tlink(timebank_tlink)

        slinks = list(soup.find_all('slink'))
        for slink in slinks:
            timebank_slink = TimebankSlink.from_bs_obj(
                slink, timebank_document
            )
            timebank_document.add_slink(timebank_slink)

        alinks = list(soup.find_all('alink'))
        for alink in alinks:
            timebank_alink = TimebankAlink.from_bs_obj(
                alink, timebank_document
            )
            timebank_document.add_alink(timebank_alink)

        makeinstances = list(soup.find_all('makeinstance'))
        for makeinstance in makeinstances:
            timebank_makeinstance = MakeInstance.from_bs_obj(
                makeinstance, timebank_document
            )
            timebank_document.add_make_instance(timebank_makeinstance)

        return timebank_document

    def to_dict(self):
        return {
            'timebank_sentences': [i.to_dict() for i in self.timebank_sentences()],  # noqa
            'slinks': [i.to_dict() for i in self.slinks()],  # noqa
            'tlinks': [i.to_dict() for i in self.tlinks()],  # noqa
            'alinks': [i.to_dict() for i in self.alinks()],  # noqa
            'make_instances': [i.to_dict() for i in self.make_instances()],  # noqa
        }
