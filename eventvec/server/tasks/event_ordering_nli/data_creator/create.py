import random
from collections import defaultdict

from eventvec.server.tasks.event_ordering_nli.datamodel.relationship import EventRelationship, parameters, parameter_name
from eventvec.server.tasks.event_ordering_nli.datamodel.event import Event



relationship_types = ['before', 'after', 'simultaneous']
test_event_place_names = [
    'Silver Crest',
    'Ironhold Pass',
    'Shadow Ridge',
    'Thunder Valley',
    'Ember Plains',
    'Raven Watch',
    'Frostgate',
    'Golden Field',
    'Misty Heights',
    'Stormforge',
    'Crimson Shore',
    'Darkwater Keep',
    'Sunfire Canyon',
    'Blackstone Crag',
    'Vipers Hollow',
    'Emberfall',
    'Wolfs Den',
    'Dragonspire Peak',
    'Silent Meadows',
    'Ironclad Bastion',
    'Windscar Ridge',
    'Serpent Pass',
    'Steelshade Valley',
    'Ashen Grove',
    'Dreadmoor',
    'Bloodthorn Keep',
    'Ravens Perch',
    'Hollowcrest',
    'Stonegate',
    'Ironhelm Hill',
    'Frostmere',
    'Shadowfen',
    'Grimwatch Tower',
    'Thundercliff',
    'Crimson Vale',
    'Sunken Reach',
    'Stormcliff',
    'Nightshade Ridge',
    'Shadowspire',
    'Dreadhold',
    'Blazing Hearth',
    'Wraithwood',
    'Frostveil',
    'Blackrock Gorge',
    'Bloodmoon Keep',
    'Silverpine',
    'Ironpeak',
    'Ravenloft',
    'Emberwatch',
    'Twilight Glade',
    'Wolfbane Tower',
    'Frostfang Pass',
    'Whispering Pines',
    'Shattered Plains',
    'Ironveil',
    'Firestone Ridge',
    'Thornwood Keep',
    'Dawnforge',
    'Stormhold',
    'Vulture Reach',
]

train_event_place_names = [
    "Obsidian Hollow",
    "Stormvale",
    "Ravenspire",
    "Ironwood Keep",
    "Shadowbrook",
    "Duskwatch Tower",
    "Blightmoor",
    "Thundershade Pass",
    "Emberglow Valley",
    "Frostwind Crag",
    "Nightfall Ridge",
    "Drake's Hollow",
    "Ashenvale",
    "Darkreach",
    "Bloodspire Keep",
    "Blackthorn Vale",
    "Whispering Hollow",
    "Silent Ridge",
    "Cinderfall",
    "Grimshade Bastion",
    "Shadowgate",
    "Dragonfang Pass",
    "Silverbrook",
    "Stormglen",
    "Daggerfall Heights",
    "Hollowshade",
    "Moonlit Crag",
    "Onyx Hollow",
    "Blazing Peak",
    "Wraithspire",
    "Thornvale",
    "Ironwatch Keep",
    "Sunscorch Canyon",
    "Frostshade Valley",
    "Stormbreaker Cliffs",
    "Duskwood",
    "Ebonridge",
    "Crimson Hollow",
    "Blacksteel Keep",
    "Ravenshadow",
    "Emberreach",
    "Ironthorn Pass",
    "Daggercliff",
    "Silent Hollow",
    "Thunderpeak",
    "Frostburn Vale",
    "Shadewatch",
    "Silverthorn",
    "Wolfsbane Hollow",
    "Duskridge"
  ]

test_event_names = [
    'Battle of Silver Crest',
    'Siege of Ironhold Pass',
    'Clash at Shadow Ridge',
    'Engagement of Thunder Valley',
    'Battle of Ember Plains',
    'Conflict at Raven Watch',
    'Assault on Frostgate',
    'Skirmish of the Golden Field',
    'Encounter at Misty Heights',
    'Struggle of the Stormforge',
    'Battle of Crimson Shore',
    'Siege of Darkwater Keep',
    'Ambush at Sunfire Canyon',
    'Engagement of Blackstone Crag',
    'Conflict at Vipers Hollow',
    'Assault on Emberfall',
    'Clash at Wolfs Den',
    'Battle of Dragonspire Peak',
    'Skirmish of Silent Meadows',
    'Raid on Ironclad Bastion',
    'Struggle at Windscar Ridge',
    'Engagement of Serpent Pass',
    'Conflict at Steelshade Valley',
    'Battle of Ashen Grove',
    'Ambush at Dreadmoor',
    'Siege of Bloodthorn Keep',
    'Skirmish at Ravens Perch',
    'Encounter at Hollowcrest',
    'Assault on Stonegate',
    'Clash at Ironhelm Hill',
    'Battle of Frostmere',
    'Engagement at Shadowfen',
    'Conflict at Grimwatch Tower',
    'Raid on Thundercliff',
    'Struggle at Crimson Vale',
    'Skirmish at Sunken Reach',
    'Siege of Stormcliff',
    'Assault on Nightshade Ridge',
    'Battle of Shadowspire',
    'Engagement of Dreadhold',
    'Clash at Blazing Hearth',
    'Conflict at Wraithwood',
    'Encounter at Frostveil',
    'Skirmish at Blackrock Gorge',
    'Battle of Bloodmoon Keep',
    'Engagement at Silverpine',
    'Siege of Ironpeak',
    'Raid on Ravenloft',
    'Struggle at Emberwatch',
    'Skirmish at Twilight Glade',
    'Assault on Wolfbane Tower',
    'Battle of Frostfang Pass',
    'Clash at Whispering Pines',
    'Engagement of Shattered Plains',
    'Conflict at Ironveil',
    'Raid on Firestone Ridge',
    'Siege of Thornwood Keep',
    'Encounter at Dawnforge',
    'Struggle of Stormhold',
    'Skirmish at Vulture Reach',
]

train_event_names = [
    "Battle of Crimson Hollow",
    "Siege of Stormveil Keep",
    "Clash at Dragon’s Roost",
    "Engagement of Shattered Hills",
    "Conflict at Blackbriar Wood",
    "Assault on Frostfire Pass",
    "Skirmish of the Starfall Plains",
    "Encounter at Ghostwood Ridge",
    "Struggle of Emberhold",
    "Battle of Blightwood",
    "Siege of Moonstone Citadel",
    "Ambush at Coldreach",
    "Raid on Granite Spire",
    "Engagement at Bloodstone Crossing",
    "Conflict at Frostveil Crag",
    "Clash at the Thunderclap Bridge",
    "Battle of Thornspire",
    "Skirmish at Ravenshade",
    "Assault on the Diremarch",
    "Struggle at Sunwatch Valley",
    "Encounter at Stormwatch Keep",
    "Battle of Grimstone Hollow",
    "Siege of Ironvale Bastion",
    "Clash at the Twilight Ruins",
    "Engagement of Thornclaw Pass",
    "Conflict at the Everpeak",
    "Skirmish at Wraithfall",
    "Battle of Shatterstone Plateau",
    "Ambush at Sunspire Bluff",
    "Assault on Hollowspire",
    "Raid on Frostpeak Summit",
    "Struggle at Ironclaw Ridge",
    "Encounter at Whisperwind Vale",
    "Battle of the Starforge",
    "Siege of Wolfhaven",
    "Engagement at Grimwatch Keep",
    "Clash at Shadowfen Marsh",
    "Conflict at Frostspire",
    "Skirmish at Thunderhill",
    "Assault on Coldspire Hold",
    "Struggle of Ironshade Ridge",
    "Raid on Ravencrest Keep",
    "Encounter at Frostbloom Fields",
    "Battle of Dreadhelm",
    "Siege of Everwinter Fortress",
    "Clash at Stormspire Peak",
    "Engagement of Duskwatch Hill",
    "Conflict at Embercliff Ridge",
    "Skirmish at Stormfire Caverns",
    "Assault on Blackthorn Fortress",
    "Struggle at Frostglade Pass",
    "Encounter at Moonshadow Keep",
    "Battle of Brimstone Hollow",
    "Siege of Iceclaw Citadel",
    "Ambush at Silverpine Pass",
    "Raid on Thunderfrost Ridge",
    "Engagement at Stormpeak Spire",
    "Clash at Bloodthorn Plains",
    "Conflict at Shadewood Crag",
    "Skirmish at Winterfell Keep",
    "Assault on Greycliff Bastion",
    "Struggle at Dawnfire Pass",
    "Encounter at Sunshadow Ridge",
    "Battle of Direfall",
    "Siege of Frostglen Keep",
    "Clash at Grimwatch Plateau",
    "Engagement of Bloodfrost Ridge",
    "Conflict at Nightfall Vale",
    "Skirmish at Ashenridge",
    "Assault on Blackspire Hold",
    "Struggle at Frostmourne Keep",
    "Raid on Emberstone Fortress",
    "Encounter at Thunderforge",
    "Battle of Ironclaw Bastion",
    "Siege of Frostwolf Citadel",
    "Clash at the Obsidian Ridge",
    "Engagement of Bloodspire Keep",
    "Conflict at Windscar Summit",
    "Skirmish at the Starspire",
    "Assault on Silvercrest Tower",
    "Struggle at Frostthorn Ridge",
    "Encounter at Dreadclaw Keep",
    "Battle of the Direforge",
    "Siege of Stonefrost Bastion",
    "Clash at Blackwatch Hill",
    "Engagement of Ravenscar Ridge",
    "Conflict at Ironwind Pass",
    "Skirmish at the Frostfire Citadel",
    "Assault on Moonspire Peak",
    "Struggle at Emberfall Tower",
    "Raid on Shadowspire Bastion",
    "Encounter at Windfall Pass",
    "Battle of Frostreach Keep",
    "Siege of Coldthorn Citadel",
    "Clash at Ironclad Hill",
    "Engagement of Silverwatch Ridge",
    "Conflict at Sunfire Summit",
    "Skirmish at Blackthorn Ridge",
    "Assault on Thunderclaw Hold"
]


simple_events_train_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'apple', 'sea', 'frog', 'item', 'car', 'boy', 'girl', 'hat', 'imp', 'joker', 'kite', 'lion', 'man', 'nectar', 'orange', 'pillow', 'queen', 'ram', 'sow', 'time', 'under', 'van', 'water', 'winter', 'yellow', 'town'
]

simple_events_test_names = [
    'round', 'plod', 'fresh', 'shirt', 'brook', 'odd', 'tail', 'harm', 'same', 'dry', 'fine', 'drop', 'card', 'class', 'drink', 'hill', 'deal', 'home', 'ask', 'apart', 'smell', 'read', 'must', 'test', 'vest', 'clad', 'add', 'belt', 'thank', 'acid', 'band', 'chalk', 'debt', 'egg', 'fear', 'goat', 'head', 'iron', 'judge', 'knife', 'linen', 'milk', 'neck', 'other', 'pipe', 'rain'
]

if parameters[parameter_name]['random_seed']:
    random.seed(0)

class Creator:
    def __init__(self):
        self._events = []
        self._relationships = []

    def create_random_relationship(self):
        event_1 = random.choice(self._events)
        point_1 = random.choice(event_1.points())
        event_2 = random.choice(list(set(self._events) - {event_1}))
        point_2 = random.choice(event_2.points())
        relationship_type = random.choice(parameters[parameter_name]['relationship_types'])
        self.create_relationship(point_1, point_2, relationship_type)

    def create_relationship(self, event_point1, event_point2, relationship_type):
        relationship = EventRelationship()
        relationship.set_relationship_type(relationship_type)
        relationship.set_event_point_1(event_point1)
        relationship.set_event_point_2(event_point2)
        event_point1.add_relationship(relationship)
        event_point2.add_relationship(relationship)
        self._relationships.append(relationship)
        return relationship

    def create_event(self, event_name=None):
        all_used_event_names = [e.event_name() for e in self._events]
        event = Event()
        if parameters[parameter_name]['names'] == 'train_event_names':
            remaining_names = list(set(simple_events_train_names) - set(all_used_event_names))
        else:
            remaining_names = list(set(simple_events_test_names) - set(all_used_event_names))
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
        s2_s1 = self.find_event_point_1_before_event_point_2(event2.start_point(), event1.end_point())
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