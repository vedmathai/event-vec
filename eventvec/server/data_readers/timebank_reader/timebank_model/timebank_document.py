from bs4 import BeautifulSoup


from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_sentence import TimebankSentence  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_tlink import TimebankTlink  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_slink import TimebankSlink  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_alink import TimebankAlink  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_makeinstance import MakeInstance  # noqa


class TimebankDocument:
    def __init__(self):
        self._tlinks = []
        self._slinks = []
        self._alinks = []
        self._timebank_sentences = []
        self._make_instances = []
        self._eid2event = {}

    def slinks(self):
        return self._slinks

    def add_slink(self, slink):
        self._tlinks.append(slink)

    def set_slinks(self, slinks):
        self._slinks = slinks

    def alinks(self):
        return self._alinks

    def add_alink(self, alink):
        self._alinks.append(alink)

    def set_alinks(self, alinks):
        self._alinks = alinks

    def tlinks(self):
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

    def eid2event(self, eid):
        return self._eid2event[eid]

    def eids(self):
        return list(self._eid2event)

    def eid2sentence(self):
        pass

    @staticmethod
    def from_xml(document):
        timebank_document = TimebankDocument()
        soup = BeautifulSoup(document, 'lxml')

        sentences = list(soup.find_all('s'))
        for s in sentences:
            timebank_sentence = TimebankSentence.from_bs_obj(
                s, timebank_document
            )  
            timebank_document.add_timebank_sentence(timebank_sentence)

        tlinks = list(soup.find_all('tlink'))
        for tlink in tlinks:
            timebank_tlink = TimebankTlink.from_bs_obj(tlink)
            timebank_document.add_tlink(timebank_tlink)

        slinks = list(soup.find_all('slink'))
        for slink in slinks:
            timebank_slink = TimebankSlink.from_bs_obj(slink)
            timebank_document.add_slink(timebank_slink)

        alinks = list(soup.find_all('alink'))
        for alink in alinks:
            timebank_alink = TimebankAlink.from_bs_obj(alink)
            timebank_document.add_alink(timebank_alink)

        makeinstances = list(soup.find_all('makeinstance'))
        for makeinstance in makeinstances:
            timebank_makeinstance = MakeInstance.from_bs_obj(makeinstance)
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