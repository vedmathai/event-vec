

class TimebankTlink:
    def __init__(self):
        self._lid = None
        self._rel_type = None
        self._event_instance_id = None
        self._related_to_event_instance = None

    def lid(self):
        return self._lid

    def rel_type(self):
        return self._rel_type

    def event_instance_id(self):
        return self._event_instance_id

    def related_to_event_instance(self):
        return self._related_to_event_instance

    def set_lid(self, lid):
        self._lid = lid

    def set_rel_type(self, rel_type):
        self._rel_type = rel_type

    def set_event_instance_id(self, event_instance_id):
        self._event_instance_id = event_instance_id

    def set_related_to_event_instance(self, related_to_event_instance):
        self._related_to_event_instance = related_to_event_instance

    @staticmethod
    def from_bs_obj(timex):
        timebank_tlink = TimebankTlink()
        timebank_tlink.set_lid(timex.attrs.get('lid'))
        timebank_tlink.set_rel_type(timex.attrs.get('rel_type'))
        timebank_tlink.set_event_instance_id(timex.attrs.get('event_instance_id'))  # noqa
        timebank_tlink.set_related_to_event_instance(timex.attrs.get('related_to_event_instance'))  # noqa
        return timebank_tlink

    def to_dict(self):
        return {
            'object_type': 'timebank_tlink',
            'lid': self.lid(),
            'rel_type': self.rel_type(),
            'event_instance_id': self.event_instance_id(),
            'related_to_event_instance': self.related_to_event_instance(),
        }
