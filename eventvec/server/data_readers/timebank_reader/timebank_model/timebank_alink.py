

class TimebankAlink:
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
    def from_bs_obj(alink):
        timebank_alink = TimebankAlink()
        timebank_alink.set_lid(alink.attrs.get('lid'))
        timebank_alink.set_rel_type(alink.attrs.get('rel_type'))
        timebank_alink.set_event_instance_id(alink.attrs.get('event_instance_id'))  # noqa
        timebank_alink.set_related_to_event_instance(alink.attrs.get('related_to_event_instance'))  # noqa
        return timebank_alink

    def to_dict(self):
        return {
            'object_type': 'timebank_alink',
            'lid': self.lid(),
            'rel_type': self.rel_type(),
            'event_instance_id': self.event_instance_id(),
            'related_to_event_instance': self.related_to_event_instance(),
        }
