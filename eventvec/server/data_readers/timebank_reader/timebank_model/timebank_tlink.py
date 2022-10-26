

class TimebankTlink:
    def __init__(self):
        self._lid = None
        self._rel_type = None
        self._time_id = None
        self._event_instance_id = None
        self._related_to_time = None
        self._related_to_event_instance = None
        self._tlink_type = None

    def lid(self):
        return self._lid

    def rel_type(self):
        return self._rel_type

    def tlink_type(self):
        return self._tlink_type

    def event_instance_id(self):
        return self._event_instance_id

    def time_id(self):
        return self._time_id

    def related_to_time(self):
        return self._related_to_time

    def related_to_event_instance(self):
        return self._related_to_event_instance

    def set_lid(self, lid):
        self._lid = lid

    def set_rel_type(self, rel_type):
        self._rel_type = rel_type

    def set_tlink_type(self, tlink_type):
        self._tlink_type = tlink_type

    def set_time_id(self, time_id):
        self._time_id = time_id

    def set_event_instance_id(self, event_instance_id):
        self._event_instance_id = event_instance_id

    def set_related_to_time(self, related_to_time):
        self._related_to_time = related_to_time

    def set_related_to_event_instance(self, related_to_event_instance):
        self._related_to_event_instance = related_to_event_instance

    @staticmethod
    def from_bs_obj(tlink, timebank_document):
        timebank_tlink = TimebankTlink()
        timebank_tlink.set_lid(tlink.attrs.get('lid'))
        timebank_tlink.set_rel_type(tlink.attrs.get('reltype'))
        set_fns = {
            'e': timebank_tlink.set_event_instance_id,
            't': timebank_tlink.set_time_id,
            '2e': timebank_tlink.set_related_to_event_instance,
            '2t': timebank_tlink.set_related_to_time,
        }
        attrs2type = {
            ('eventinstanceid', 'relatedtoeventinstance'): 'e2e',
            ('eventinstanceid', 'relatedtotime'): 'e2t',
            ('timeid', 'relatedtoeventinstance'): 't2e',
            ('timeid', 'relatedtotime'): 't2t',
        }
        add_fns = {
            't': (timebank_document.add_event_instance_id2tlink, timebank_tlink.event_instance_id),  # noqa
            'e': (timebank_document.add_time_id2tlink, timebank_tlink.time_id)  # noqa
        }
        for attrs in attrs2type:
            if attrs[0] in tlink.attrs and attrs[1] in tlink.attrs:
                attr0 = attrs[0]
                attr1 = attrs[1]
                _type = attrs2type[(attrs[0], attrs[1])]
                timebank_tlink.set_tlink_type(_type)
        function1 = set_fns[_type[0]]
        function2 = set_fns[_type[1:]]
        function1(tlink.attrs.get(attr0))
        function2(tlink.attrs.get(attr1))
        add_fn, id = add_fns[_type[0]]
        add_fn(id, timebank_tlink)
        return timebank_tlink

    def to_dict(self):
        return {
            'object_type': 'timebank_tlink',
            'lid': self.lid(),
            'rel_type': self.rel_type(),
            'event_instance_id': self.event_instance_id(),
            'related_to_event_instance': self.related_to_event_instance(),
        }
