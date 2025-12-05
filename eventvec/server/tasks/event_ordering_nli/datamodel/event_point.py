import uuid

from eventvec.server.tasks.event_ordering_nli.data_creator.parameters import parameters

domain2template = {
    'temporal': {
        'start': 'start of {}',
        'end': 'end of {}',
    },
    'spatial': {
        'start': 'eastern edge of {}',
        'end': 'western edge of {}',
    },
    'logical': {
        'start': '{}1',
        'end': '{}2',
    }
}

class EventPoint():
    def __init__(self, parameter):
        self._id = 'event_point_' + str(uuid.uuid4())
        self._event = None
        self._relationships = []
        self._is_start = True
        self._parameter_name = parameter

    def event(self):
        return self._event
    
    def relationships(self):
        return self._relationships
    
    def is_start(self):
        return self._is_start
    
    def set_event(self, event):
        self._event = event

    def add_relationship(self, relationship):
        self._relationships.append(relationship)

    def set_relationships(self, relationships):
        self._relationships = relationships

    def set_is_start(self, is_start):
        self._is_start = is_start

    def to_dict(self):
        return {
            'event': self._event,
            'relationships': self._relationships,
            'is_start': self._is_start
        }
    
    def from_dict(self, data):
        self._event = data['event']
        self._relationships = data['relationships']
        self._is_start = data['is_start']

    def __hash__(self) -> int:
        return hash(self._id)

    def __repr__(self):
        domain = parameters[self._parameter_name]['domain']
        if self.is_start():
            return domain2template[domain]['start'].format(self.event().event_name())
        else:
            return domain2template[domain]['end'].format(self.event().event_name())
