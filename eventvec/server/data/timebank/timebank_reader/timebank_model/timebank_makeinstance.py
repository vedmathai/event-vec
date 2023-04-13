class MakeInstance:
    def __init__(self):
        self._event_id = None
        self._eiid = None
        self._tense = None
        self._aspect = None
        self._polarity = None
        self._pos = None
        self._modality = None

    def event_id(self):
        return self._event_id

    def eiid(self):
        return self._eiid

    def tense(self):
        return self._tense

    def aspect(self):
        return self._aspect

    def polarity(self):
        return self._polarity

    def pos(self):
        return self._pos

    def modality(self):
        return self._modality

    def set_event_id(self, event_id):
        self._event_id = event_id

    def set_eiid(self, eiid):
        self._eiid = eiid

    def set_tense(self, tense):
        self._tense = tense

    def set_aspect(self, aspect):
        self._aspect = aspect

    def set_polarity(self, polarity):
        self._polarity = polarity

    def set_pos(self, pos):
        self._pos = pos

    def set_modality(self, modality):
        self._modality = modality

    @staticmethod
    def from_bs_obj(bs_obj, timebank_document):
        timebank_makeinstance = MakeInstance()
        timebank_makeinstance.set_event_id(bs_obj.attrs.get('eventid'))
        timebank_makeinstance.set_eiid(bs_obj.attrs.get('eiid'))
        timebank_makeinstance.set_tense(bs_obj.attrs.get('tense'))
        timebank_makeinstance.set_aspect(bs_obj.attrs.get('aspect'))
        timebank_makeinstance.set_polarity(bs_obj.attrs.get('polarity'))
        timebank_makeinstance.set_pos(bs_obj.attrs.get('pos'))
        timebank_makeinstance.set_modality(bs_obj.attrs.get('modality'))
        timebank_document.add_event_id2make_instance(
            timebank_makeinstance.event_id(), timebank_makeinstance
        )
        timebank_document.add_event_instance_id2make_instance(
            timebank_makeinstance.eiid(), timebank_makeinstance
        )

        return timebank_makeinstance

    def to_dict(self):
        return {
            'object_type': 'makeinstance',
            'event_id': self.event_id(),
            'eiid': self.eiid(),
            'tense': self.tense(),
            'aspect': self.aspect(),
            'polarity': self.polarity(),
            'pos': self.pos(),
            'modality': self.modality(),
        }
