import uuid
import random

from eventvec.server.tasks.event_ordering_nli.datamodel.templates import templates_dict
from eventvec.server.tasks.event_ordering_nli.data_creator.parameters import parameters

class EventRelationship():
    def __init__(self, parameter_name):
        self._id = 'relationship_' + str(uuid.uuid4())
        self._relationship_type = None
        self._event_point_1 = None
        self._event_point_2 = None
        self._parameter_name = parameter_name
        self._train_key = parameters[self._parameter_name]['templates'] 
        self._domain_key = parameters[self._parameter_name]['domain']

    def id(self):
        return self._id
    
    def relationship_type(self):
        return self._relationship_type
    
    def event_point_1(self):
        return self._event_point_1
    
    def event_point_2(self):
        return self._event_point_2
    
    def other_point(self, point):
        if point == self._event_point_1:
            return self._event_point_2
        elif point == self._event_point_2:
            return self._event_point_1
    
    def set_id(self, id):
        self._id = id

    def set_relationship_type(self, relationship_type):
        self._relationship_type = relationship_type

    def set_event_point_1(self, event_point_1):
        self._event_point_1 = event_point_1
    
    def set_event_point_2(self, _event_point_2):
        self._event_point_2 = _event_point_2

    def to_dict(self):
        return {
            'id': self._id,
            'relationship_type': self._relationship_type,
            'event_point_1': self._event_point_1,
            'event_point_2': self._event_point_2,
        }
    
    def from_dict(self, data):
        self._id = data['id']
        self._relationship_type = data['relationship_type']
        self._event_point_1 = data['event_point_1']
        self._event_point_2 = data['event_point_2']

    def __hash__(self) -> int:
        return hash(self._id)
    
    def get_after_sentence(self, event_point_1, event_point_2):
        templates = templates_dict[self._domain_key][self._train_key]['after']
        return random.choice(templates).format(A=event_point_1, B=event_point_2)
    
    def get_before_switched_sentence(self, event_point_1, event_point_2):
        templates = templates_dict[self._domain_key][self._train_key]['before']
        return random.choice(templates).format(A=event_point_2, B=event_point_1)
    
    def get_before_sentence(self, event_point_1, event_point_2):
        templates = templates_dict[self._domain_key][self._train_key]['before']
        return random.choice(templates).format(A=event_point_1, B=event_point_2)
    
    def get_simultaneous_sentence(self, event_point_1, event_point_2):
        templates = templates_dict[self._domain_key][self._train_key]['simultaneous']
        return random.choice(templates).format(A=event_point_1, B=event_point_2)

    def __str__(self):
        if self.relationship_type() == 'simultaneous':
            return self.get_simultaneous_sentence(self._event_point_1, self._event_point_2)
        elif self.relationship_type() == 'before':
            return self.get_before_sentence(self._event_point_1, self._event_point_2)
        elif self.relationship_type() == 'after':
            return self.get_after_sentence(self._event_point_1, self._event_point_2)
