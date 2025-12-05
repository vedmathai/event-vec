import random
from collections import defaultdict

from eventvec.server.tasks.event_ordering_nli.datamodel.relationship import EventRelationship
from eventvec.server.tasks.event_ordering_nli.data_creator.parameters import parameters
from eventvec.server.tasks.event_ordering_nli.datamodel.event import Event
from eventvec.server.tasks.event_ordering_nli.datamodel.event_names import event_names_dict


relationship_types = ['before', 'after', 'simultaneous']


class Creator:
    def __init__(self, parameter_name):
        self._events = []
        self._relationships = []
        self._parameter_name = parameter_name

    def create_random_relationship(self):
        event_1 = random.choice(self._events)
        point_1 = random.choice(event_1.points())
        event_2 = random.choice(list(set(self._events) - {event_1}))
        point_2 = random.choice(event_2.points())
        relationship_type = random.choice(parameters[self._parameter_name]['relationship_types'])
        self.create_relationship(point_1, point_2, relationship_type)

    def create_relationship(self, event_point1, event_point2, relationship_type):
        relationship = EventRelationship(self._parameter_name)
        relationship.set_relationship_type(relationship_type)
        relationship.set_event_point_1(event_point1)
        relationship.set_event_point_2(event_point2)
        event_point1.add_relationship(relationship)
        event_point2.add_relationship(relationship)
        self._relationships.append(relationship)
        return relationship

    def create_event(self, event_name=None):
        all_used_event_names = [e.event_name() for e in self._events]
        event = Event(self._parameter_name)
        event_names = event_names_dict[parameters[self._parameter_name]['names']]
        remaining_names = list(set(event_names) - set(all_used_event_names))
        remaining_names = sorted(remaining_names)
        self.create_relationship(event.start_point(), event.end_point(), 'before')
        if event_name is None:
            event_name = random.choice(remaining_names)
        event.set_event_name(event_name)
        self._events.append(event)
        return event

    def print_relationships(self):
        for r in self._relationships:
            print(r)

    def event_points(self):
        event_points = []
        for event in self._events:
            event_points.append(event.start_point())
            event_points.append(event.end_point())
        return event_points

    def find_earliest(self):
        earliest = self._events[0]
        relationships = self._relationships
        seen = set()
        while len(relationships) > 0 and len(set(relationships) - seen) > 0:
            remaining = list((set(relationships) - seen))
            remaining = sorted(remaining, key=lambda x: x.event_point_1().event().event_name())
            relationship = remaining.pop()
            seen.add(relationship)
            if relationship.relationship_type() == 'before':
                earliest = relationship.event_point_1().event()
                relationships = earliest.relationships()
            if relationship.relationship_type() == 'after':
                earliest = relationship.event_point_2().event()
                relationships = earliest.relationships()
            if relationship.relationship_type() == 'simultaneous' and earliest is not None:
                simultaneous = relationship.other_point(earliest).event()
                relationships += simultaneous.relationships()
        return earliest
    
    def find_total_order(self):
        events = [[e] for e in self.event_points()]
        ei1 = 0
        while ei1 < len(events) - 1:
            for event in events[ei1]:
                for relationship in event.relationships():
                    ei2 = ei1 + 1
                    while ei2 < len(events):
                        #print(events[ei1], events[ei2], relationship, relationship.relationship_type() == 'simultaneous', relationship.other_point(event), events[ei2])
                        if relationship.relationship_type() == 'simultaneous' and relationship.other_point(event) in events[ei2]:
                            e2 = events[ei2]
                            events = events[:ei2] + events[ei2+1:]
                            events[ei1] += e2
                        ei2 += 1
            ei1 += 1
        while ei1 < len(events) - 1:
            for event in events[ei1]:
                for relationship in event.relationships():
                    ei2 = ei1 + 1
                    while ei2 < len(events):
                        if relationship.relationship_type() == 'after' and relationship.event_point_2().event() in events[ei2]:
                            events[ei1], events[ei2] = events[ei2], events[ei1]
                        if relationship.relationship_type() == 'before' and relationship.event_point_1().event() in events[ei2]:
                            events[ei1], events[ei2] = events[ei2], events[ei1]
                        ei2 += 1
            ei1 += 1
        return events

    def find_event_point_1_before_event_point_2(self, event_point_1, event_point_2):
        event_points = [(event_point_1, True)]  # event point and whether simultaneous
        seen = set()
        while len(event_points) > 0:
            event_point, is_simultaneous = event_points.pop()
            if event_point in seen:
                continue
            seen.add(event_point)
            for relationship in event_point.relationships():
                if relationship.relationship_type() == 'after' and event_point == relationship.event_point_2():
                    event_points.append((relationship.event_point_1(), False))
                if relationship.relationship_type() == 'before' and event_point == relationship.event_point_1():
                    event_points.append((relationship.event_point_2(), False))
                if relationship.relationship_type() == 'simultaneous':
                    event_points.append((relationship.other_point(event_point), is_simultaneous))
                if (event_point_2, False) in event_points:
                    return True
        return False
    
    def is_simultaneous_events(self, event1, event2):
        events = [event1]
        event_points = [event1.start_point(), event1.end_point()]
        while len(events) > 0:
            event = events.pop()
            for relationship in event.start_point().relationships():
                if relationship.relationship_type() == 'simultaneous':
                    events.append(relationship.other_point(event.start_point()).event())
                    event_points.append(relationship.other_point(event.start_point()))
                if event2.start_point() in event_points and event2.end_point() in event_points:
                    return True
        return False
    
    def is_simultaneous_event_points(self, event_point1, event_point2):
        eventpoints = [event_point1]
        seen = set()
        while len(eventpoints) > 0:
            event_point = eventpoints.pop()
            seen.add(event_point)
            for relationship in event_point.relationships():
                if relationship.relationship_type() == 'simultaneous':
                    other_point = relationship.other_point(event_point)
                    if other_point not in seen:
                        eventpoints.append(relationship.other_point(event_point))
                if event_point2 in eventpoints:
                    return True
        return False

    def does_overlap_forwards(self, event1, event2):
        e1_s1 = self.find_event_point_1_before_event_point_2(event1.end_point(), event1.start_point())
        e2_s2 = self.find_event_point_1_before_event_point_2(event2.end_point(), event2.start_point())
        s2_s1 = self.find_event_point_1_before_event_point_2(event2.start_point(), event1.start_point())
        s1_s2 = self.find_event_point_1_before_event_point_2(event1.start_point(), event2.start_point())
        s2_e1 = self.find_event_point_1_before_event_point_2(event2.start_point(), event1.end_point())
        check = (not e1_s1) and (not e2_s2) and (not s2_s1 or s1_s2) and s2_e1
        return check
    
    def is_overlap_events(self, event1, event2):
        impossible = self.is_impossible_event_pair(event1, event2)
        e1_e2 = self.does_overlap_forwards(event1, event2)
        e2_e1 = self.does_overlap_forwards(event2, event1)
        return not impossible and (e1_e2 or e2_e1)
    
    def is_strictly_before(self, event1, event2):
        s1_s2 = self.find_event_point_1_before_event_point_2(event1.start_point(), event2.start_point())
        e1_s2 = self.find_event_point_1_before_event_point_2(event1.end_point(), event2.start_point())
        e1_s2_is_simultaneous = self.is_simultaneous_event_points(event1.end_point(), event2.start_point())
        check = s1_s2 & (e1_s2 or e1_s2_is_simultaneous)
        return check
    
    def is_impossible_event_pair(self, event_1, event_2):
        s1_s2 = self.is_impossible_event_points(event_1.start_point(), event_2.start_point())
        s1_e1 = self.is_impossible_event_points(event_1.start_point(), event_1.end_point())
        s1_e2 = self.is_impossible_event_points(event_1.start_point(), event_2.end_point())
        s2_e2 = self.is_impossible_event_points(event_2.start_point(), event_2.end_point())
        return s1_s2 or s1_e1 or s1_e2 or s2_e2
   
    def is_impossible_event_points(self, event_point_1, event_point_2):
        same_event_check = (event_point_1.event() == event_point_2.event()) and (event_point_1.event().start_point() == event_point_1) and (event_point_2.event().end_point() == event_point_2)
        check_forwards = self.find_event_point_1_before_event_point_2(event_point_1, event_point_2)
        check_backwards = self.find_event_point_1_before_event_point_2(event_point_2, event_point_1)
        check_simultaneous = self.is_simultaneous_event_points(event_point_1, event_point_2)
        return check_forwards and check_backwards and not check_simultaneous and not (same_event_check and check_backwards)
    
    def find_all_impossible_event_points(self):
        impossible_events = []
        event_points = list(set(self.event_points()))
        for event_point_1 in event_points:
            for event_point_2 in event_points:
                if event_point_1 == event_point_2:
                    continue
                if self.is_impossible_event_points(event_point_1, event_point_2):
                    impossible_events.append((event_point_1, event_point_2))
        return impossible_events

    def find_path_all_events(self):
        event_points = list(set(self.event_points()))
        for event_point_1 in event_points:
            for event_point_2 in event_points:
                if event_point_1 == event_point_2:
                    continue
                print('{}|{}|{}'.format(event_point_1, event_point_2, self.find_event_point_1_before_event_point_2(event_point_1, event_point_2)))

    def find_overlaps_events(self):
        for event_1 in self._events:
            for event_2 in self._events:
                if event_1 == event_2:
                    continue
                print('{}|{}|{}'.format(event_1, event_2, self.does_overlap(event_1, event_2) or self.does_overlap(event_2, event_1)))

    def event2distances(self, event_1):
        events = [(event_1, 0)]
        event2distance = defaultdict(lambda: float('inf'))
        seen = set()
        while len(events) > 0:
            event, distance = events.pop(0)
            seen.add(event)
            event2distance[event] = min(event2distance[event], distance)
            for relationship in event.start_point().relationships():
                other_point = relationship.other_point(event.start_point())
                if other_point.event() not in seen:
                    events.append((other_point.event(), distance + 1))
            for relationship in event.end_point().relationships():
                other_point = relationship.other_point(event.end_point())
                if other_point.event() not in seen:
                    events.append((other_point.event(), distance + 1))
        return event2distance

    def distance_between_events(self, event_1, event_2):
        return self.event2distances(event_1)[event_2]
    
    def events2distances(self):
        events2distances = defaultdict(lambda: defaultdict(lambda: float('inf')))
        for event in self._events:
            event2distance = self.event2distances(event)
            for other_event, distance in event2distance.items():
                events2distances[event][other_event] = distance
        return events2distances
    
    def distances2events(self):
        distances2events = defaultdict(list)
        events2distances = self.events2distances()
        for event_1 in events2distances:
            for event_2 in events2distances[event_1]:
                distance = events2distances[event_1][event_2]
                distances2events[distance].append((event_1, event_2))
        return distances2events
    

    def sort_relationships(self):
        for relationship_1i, relationship_1 in enumerate(self._relationships):
            for relationship_2i, relationship_2 in enumerate(self._relationships):
                if relationship_1.relationship_type() in ['before', 'simultaneous']:
                    r1_e1 = relationship_1.event_point_1()
                else:
                    r1_e1 = relationship_1.event_point_2()
                if relationship_2.relationship_type() in ['before', 'simultaneous']:
                    r2_e1 = relationship_2.event_point_1()
                else:
                    r2_e1 = relationship_2.event_point_2()
                if self.find_event_point_1_before_event_point_2(r2_e1, r1_e1):
                    self._relationships[relationship_1i], self._relationships[relationship_2i] = self._relationships[relationship_2i], self._relationships[relationship_1i]


if __name__ == '__main__':
    creator = Creator()
    creator.create_event()
    for r in creator._relationships:
        print(r)