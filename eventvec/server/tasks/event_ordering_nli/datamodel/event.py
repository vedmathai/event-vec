import uuid

from eventvec.server.tasks.event_ordering_nli.datamodel.event_point import EventPoint


class Event():
    def __init__(self):
        self._id = 'event_' + str(uuid.uuid4())
        self._event_name = None
        self._start_point = None
        self._end_point = None

    def id(self):
        return self._id
    
    def event_name(self):
        return self._event_name
    
    def points(self):
        return [self.start_point(), self.end_point()]
    
    def relationships(self):
        return self.start_point().relationships() + self.end_point().relationships()
    
    def start_point(self):
        if self._start_point is None:
            self._start_point = EventPoint()
            self._start_point.set_is_start(True)
            self._start_point.set_event(self)
        return self._start_point
    
    def end_point(self):
        if self._end_point is None:
            self._end_point = EventPoint()
            self._end_point.set_is_start(False)
            self._end_point.set_event(self)
        return self._end_point
    
    def add_relationship(self, event_point, relationship):
        if event_point == self._start_point:
            self._start_point.add_relationship(relationship)
        elif event_point == self._end_point:
            self._end_point.add_relationship(relationship)
        else:
            raise ValueError('Event point not found')
    
    def set_id(self, id):
        self._id = id

    def set_event_name(self, event_name):
        self._event_name = event_name

    def set_start_point(self, start_point):
        self._start_point = start_point

    def set_end_point(self, end_point):
        self._end_point = end_point

    def to_dict(self):
        return {
            'id': self._id,
            'event_name': self._event_name,
            'start_point': self._start_point,
            'end_point': self._end_point
        }
    
    def from_dict(self, data):
        self._id = data['id']
        self._event_name = data['event_name']
        self._start_point = data['start_point']
        self._end_point = data['end_point']

    def __repr__(self):
        return self._event_name
