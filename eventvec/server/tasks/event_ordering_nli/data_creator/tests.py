import unittest

from eventvec.server.tasks.event_ordering_nli.data_creator.create import Creator
from eventvec.server.tasks.event_ordering_nli.datamodel.relationship import EventRelationship
from eventvec.server.tasks.event_ordering_nli.datamodel.event import Event
from eventvec.server.tasks.event_ordering_nli.datamodel.event_point import EventPoint


class TestEventsCreator(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._creator = Creator()

    def create_event(self, event_name):
        return self._creator.create_event(event_name)

    def test_simple_forwards(self):
        e1 = self.create_event('e1')
        e2 = self.create_event('e2')
        r1 = self._creator.create_relationship(e1.start_point(), e2.end_point(), 'before')
        e1_e2 = self._creator.find_event_point_1_before_event_point_2(e1.start_point(), e2.end_point())
        self.assertTrue(e1_e2)

        e2_e1 = self._creator.find_event_point_1_before_event_point_2(e2.end_point(), e1.start_point())
        self.assertFalse(e2_e1)

    def test_closure_forwards(self):
        e1 = self.create_event('e1')
        e2 = self.create_event('e2')
        e3 = self.create_event('e3')
        r1 = self._creator.create_relationship(e1.start_point(), e2.end_point(), 'before')
        r2 = self._creator.create_relationship(e2.end_point(), e3.end_point(), 'before')
        e1_e3 = self._creator.find_event_point_1_before_event_point_2(e1.start_point(), e3.end_point())
        self.assertTrue(e1_e3)
        e3_e1 = self._creator.find_event_point_1_before_event_point_2(e3.end_point(), e1.start_point())
        self.assertFalse(e3_e1)

    def test_simultaneous(self):
        e1 = self.create_event('e1')
        e2 = self.create_event('e2')
        e3 = self.create_event('e3')
        r1 = self._creator.create_relationship(e1.start_point(), e2.start_point(), 'simultaneous')
        r2 = self._creator.create_relationship(e2.start_point(), e3.start_point(), 'simultaneous')
        e1_e3 = self._creator.is_simultaneous_event_points(e1.start_point(), e3.start_point())
        self.assertTrue(e1_e3)
        e1_e3 = self._creator.find_event_point_1_before_event_point_2(e1.start_point(), e3.start_point())
        self.assertFalse(e1_e3)

    def test_impossible(self):
        e1 = self.create_event('e1')
        e2 = self.create_event('e2')
        e3 = self.create_event('e3')
        r1 = self._creator.create_relationship(e1.start_point(), e2.end_point(), 'before')
        r2 = self._creator.create_relationship(e2.end_point(), e3.end_point(), 'before')
        e1_e2 = self._creator.is_impossible_event_points(e1.start_point(), e2.end_point())
        self.assertFalse(e1_e2)

        r3 = self._creator.create_relationship(e3.end_point(), e1.start_point(), 'before')
        e1_e3 = self._creator.is_impossible_event_points(e1.start_point(), e3.end_point())
        self.assertTrue(e1_e3)

        impossible_events = self._creator.find_all_impossible_event_points()
        self.assertEqual(len(impossible_events), 6)

    def test_impossible_event_pair(self):
        e1 = self.create_event('e1')
        e2 = self.create_event('e2')
        r1 = self._creator.create_relationship(e1.start_point(), e2.start_point(), 'before')
        r2 = self._creator.create_relationship(e1.start_point(), e2.end_point(), 'after')

        e1_e2 = self._creator.is_impossible_event_pair(e1, e2)
        self.assertTrue(e1_e2)

    def test_overlap(self):
        e1 = self.create_event('e1')
        e2 = self.create_event('e2')
        e3 = self.create_event('e3')
        r1 = self._creator.create_relationship(e1.start_point(), e2.start_point(), 'before')
        r2 = self._creator.create_relationship(e1.end_point(), e2.start_point(), 'before')
        e1_e2 = self._creator.is_overlap_events(e1, e2)
        self.assertFalse(e1_e2)

        r3 = self._creator.create_relationship(e1.start_point(), e3.start_point(), 'before')
        r4 = self._creator.create_relationship(e1.start_point(), e3.end_point(), 'before')
        r5 = self._creator.create_relationship(e1.end_point(), e3.start_point(), 'after')
        e1_e3 = self._creator.is_overlap_events(e1, e3)
        self.assertTrue(e1_e3)
        
        e4 = self.create_event('e4')

        r6 = self._creator.create_relationship(e1.start_point(), e4.start_point(), 'after')
        r7 = self._creator.create_relationship(e4.start_point(), e3.start_point(), 'after')

        e1_e3 = self._creator.is_overlap_events(e1, e3)
        self.assertFalse(e1_e3)

    def test_distance_between_events(self):
        e1 = self.create_event('e1')
        e2 = self.create_event('e2')
        e3 = self.create_event('e3')
        e4 = self.create_event('e4')

        r1 = self._creator.create_relationship(e1.start_point(), e2.start_point(), 'before')
        r2 = self._creator.create_relationship(e2.end_point(), e3.start_point(), 'before')
        r3 = self._creator.create_relationship(e4.start_point(), e3.start_point(), 'before')

        dist = self._creator.distance_between_events(e1, e4)
        self.assertEqual(dist, 3)

        r1 = self._creator.create_relationship(e1.start_point(), e4.start_point(), 'before')
        dist = self._creator.distance_between_events(e1, e4)
        self.assertEqual(dist, 1)

if __name__ == '__main__':
    unittest.main()