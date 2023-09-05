
class TimebankSlink:
    def __init__(self):
        self._lid = None
        self._rel_type = None
        self._event_instance_id = None
        self._subordinated_event_instance = None

    def lid(self):
        return self._lid

    def rel_type(self):
        return self._rel_type

    def event_instance_id(self):
        return self._event_instance_id

    def subordinated_event_instance(self):
        return self._subordinated_event_instance

    def set_lid(self, lid):
        self._lid = lid

    def set_rel_type(self, rel_type):
        self._rel_type = rel_type

    def set_event_instance_id(self, event_instance_id):
        self._event_instance_id = event_instance_id

    def set_subordinated_event_instance(self, subordinated_event_instance):
        self._subordinated_event_instance = subordinated_event_instance

    @staticmethod
    def from_bs_obj(slink, timebank_document):
        timebank_slink = TimebankSlink()
        timebank_slink.set_lid(slink.attrs.get('lid'))
        timebank_slink.set_rel_type(slink.attrs.get('reltype'))
        timebank_slink.set_event_instance_id(slink.attrs.get('eventinstanceid'))  # noqa
        timebank_slink.set_subordinated_event_instance(slink.attrs.get('subordinatedeventinstance'))  # noqa
        timebank_document.add_event_instance_id2slink(
            timebank_slink.event_instance_id(), timebank_slink
        )
        return timebank_slink

    def to_dict(self):
        return {
            'object_type': 'timebank_slink',
            'lid': self.lid(),
            'rel_type': self.rel_type(),
            'event_instance_id': self.event_instance_id(),
            'subordinated_event_instance': self.subordinated_event_instance(),
        }
