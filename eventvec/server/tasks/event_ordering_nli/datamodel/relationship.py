import uuid
import random

parameters = {
    'temporal_nli_test': {
        'filename': 'temporal_nli_test.csv',
        'random_seed': True,
        'names': 'test_event_names',
        'templates': 'test',
        'relationship_types': ['before', 'after', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_simple_event_names_train': {
        'filename': 'temporal_nli_simple_event_names_train.csv',
        'random_seed': False,
        'names': 'train_event_names',
        'templates': 'test',
        'relationship_types': ['before', 'after', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_simple_event_names_test': {
        'filename': 'temporal_nli_simple_event_names_test.csv',
        'random_seed': False,
        'names': 'test_event_names',
        'templates': 'test',
        'relationship_types': ['before', 'after', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_all_diff_train': {
        'filename': 'temporal_nli_all_diff_train.csv',
        'random_seed': False,
        'names': 'train_event_names',
        'templates': 'train',
        'relationship_types': ['before', 'after', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_same_english_train': {
        'filename': 'temporal_nli_same_english_train.csv',
        'random_seed': False,
        'names': 'train_event_names',
        'templates': 'test',
        'relationship_types': ['before', 'after', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_same_names': {
        'filename': 'temporal_nli_same_names.csv',
        'random_seed': False,
        'names': 'test_event_names',
        'templates': 'train',
        'relationship_types': ['before', 'after', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_same_structures_train': {
        'filename': 'temporal_nli_same_structures_train.csv',
        'random_seed': True,
        'names': 'train_event_names',
        'templates': 'train',
        'relationship_types': ['before', 'after', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_same_structures_and_templates_train': {
        'filename': 'temporal_nli_same_structures_and_templates_train.csv',
        'random_seed': True,
        'names': 'train_event_names',
        'templates': 'test',
        'relationship_types': ['before', 'after', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_before_after_train': {
        'filename': 'temporal_nli_before_after_train.csv',
        'random_seed': False,
        'names': 'train_event_names',
        'templates': 'train',
        'relationship_types': ['before', 'after'],
        'sort_relationships': False,
    },
    'temporal_nli_only_before_sort_train': {
        'filename': 'temporal_nli_only_before_sort_train.csv',
        'random_seed': False,
        'names': 'train_event_names',
        'templates': 'train',
        'relationship_types': ['before', 'after'],
        'sort_relationships': True,
    },
    'temporal_nli_only_before_sort_test': {
        'filename': 'temporal_nli_only_before_sort_test.csv',
        'random_seed': False,
        'names': 'test_event_names',
        'templates': 'test',
        'relationship_types': ['before', 'after'],
        'sort_relationships': True,
    },
    'temporal_nli_before_after_test': {
        'filename': 'temporal_nli_before_after_test.csv',
        'random_seed': False,
        'names': 'test_event_names',
        'templates': 'test',
        'relationship_types': ['before', 'after'],
        'sort_relationships': False,
    },
    'temporal_nli_before_simultaneous_train': {
        'filename': 'temporal_nli_before_simultaneous_train.csv',
        'random_seed': False,
        'names': 'train_event_names',
        'templates': 'train',
        'relationship_types': ['before', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_before_simultaneous_test': {
        'filename': 'temporal_nli_before_simultaneous_test.csv',
        'random_seed': False,
        'names': 'test_event_names',
        'templates': 'test',
        'relationship_types': ['before', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_after_simultaneous_train': {
        'filename': 'temporal_nli_after_simultaneous_train.csv',
        'random_seed': False,
        'names': 'train_event_names',
        'templates': 'train',
        'relationship_types': ['after', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_after_simultaneous_test': {
        'filename': 'temporal_nli_after_simultaneous_test.csv',
        'random_seed': False,
        'names': 'test_event_names',
        'templates': 'test',
        'relationship_types': ['after', 'simultaneous'],
        'sort_relationships': False,
    },
    'temporal_nli_sort_relationships_train': {
        'filename': 'temporal_nli_sort_relationships_train.csv',
        'random_seed': False,
        'names': 'train_event_names',
        'templates': 'train',
        'relationship_types': ['after', 'before', 'simultaneous'],
        'sort_relationships': True,
    },
    'temporal_nli_sort_relationships_test': {
        'filename': 'temporal_nli_sort_relationships_test.csv',
        'random_seed': False,
        'names': 'test_event_names',
        'templates': 'test',
        'relationship_types': ['after', 'before', 'simultaneous'],
        'sort_relationships': True,
    },
    'spatial_nli_test': {
        'filename': 'spatial_nli_relationships_test.csv',
        'random_seed': False,
        'names': 'test_event_names',
        'templates': 'test',
        'relationship_types': ['after', 'before', 'simultaneous'],
        'sort_relationships': False,
    },
    'spatial_nli_train': {
        'filename': 'spatial_nli_relationships_train.csv',
        'random_seed': False,
        'names': 'train_event_names',
        'templates': 'train',
        'relationship_types': ['after', 'before', 'simultaneous'],
        'sort_relationships': False,
    },
    'logical_nli_train': {
        'filename': 'logical_nli_relationships_train.csv',
        'random_seed': False,
        'names': 'train_event_names',
        'templates': 'train',
        'relationship_types': ['after', 'before', 'simultaneous'],
        'sort_relationships': False,
    },
    'logical_nli_test': {
        'filename': 'logical_nli_relationships_test.csv',
        'random_seed': False,
        'names': 'test_event_names',
        'templates': 'test',
        'relationship_types': ['after', 'before', 'simultaneous'],
        'sort_relationships': False,
    }
}

parameter_name = 'logical_nli_test'

before_train_templates = [
    "The {A} occurred before the {B}",
    "The {A} took place prior to the {B}",
    "The {A} happened earlier than the {B}",
    "The {A} preceded the {B}",
    "The {A} came before the {B}",
    "The {A} unfolded ahead of the {B}",
    "The {A} transpired before the {B}",
    "The {A} occurred in advance of the {B}",
    "The {A} happened sooner than the {B}",
    "The {A} took place earlier than the {B}"
]

after_train_templates = [
    "The {A} occurred following the {B}",
    "The {A} took place subsequent to the {B}",
    "The {A} transpired after the {B}",
    "The {A} ensued once the {B} had happened",
    "The {A} came after the {B}",
    "The {A} unfolded in the wake of the {B}",
    "The {A} happened later than the {B}",
    "The {A} succeeded the {B}",
    "The {A} occurred as a result of the {B}",
    "The {A} followed in the aftermath of the {B}"
  ]

simultaneous_train_templates = [
    "The {A} occurred simultaneously with the {B}",
    "The {A} took place at the same time as the {B}",
    "The {A} happened concurrently with {B}",
    "The {A} unfolded together with {B}",
    "The {A} coincided with the {B}",
    "The {A} happened in parallel with the {B}",
    "The {A} occurred in unison with the {B}",
    "The {A} happened at the exact same moment as {B}",
    "The {A} was synchronized with the {B}",
    "The {A} took place concurrently with {B}"
  ]

simultaneous_test_templates = [
    "The {A} happened simultaneous to the {B}",
]

after_test_templates = [
    "The {A} happened after the {B}",
]

before_test_templates = [
    "The {A} happened before the {B}",
]


simultaneous_spatial_train_templates = [
    "{A} is on the same meridian as {B}",
]

after_spatial_train_templates = [
    "{A} is eastwards of {B}",
]

before_spatial_train_templates = [
    "{A} is westwards of {B}",
]


simultaneous_spatial_test_templates = [
    "{A} is on the same longitude as {B}",
]

after_spatial_test_templates = [
    "{A} is located to the east of {B}",
]

before_spatial_test_templates = [
    "{A} is located to the west of {B}",
]

simultaneous_logical_train_templates = [
    "{A} = {B}",
]

after_logical_train_templates = [
    "{A} > {B}",
]

before_logical_train_templates = [
    "{A} < {B}",
]


if parameters[parameter_name]['random_seed']:
    random.seed(0)


class EventRelationship():
    def __init__(self):
        self._id = 'relationship_' + str(uuid.uuid4())
        self._relationship_type = None
        self._event_point_1 = None
        self._event_point_2 = None

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
        if parameters[parameter_name]['templates'] == 'train':
            templates = after_logical_train_templates
        else:
            templates = after_logical_train_templates
        return random.choice(templates).format(A=event_point_1, B=event_point_2)
    
    def get_before_switched_sentence(self, event_point_1, event_point_2):
        if parameters[parameter_name]['templates'] == 'train':
            templates = before_spatial_test_templates
        else:
            templates = before_test_templates
        return random.choice(templates).format(A=event_point_1, B=event_point_2)
    
    def get_before_sentence(self, event_point_1, event_point_2):
        if parameters[parameter_name]['templates'] == 'train':
            templates = before_logical_train_templates
        else:
            templates = before_logical_train_templates
        return random.choice(templates).format(A=event_point_1, B=event_point_2)
    
    def get_simultaneous_sentence(self, event_point_1, event_point_2):
        if parameters[parameter_name]['templates'] == 'train':
            templates = simultaneous_logical_train_templates
        else:
            templates = simultaneous_logical_train_templates
        return random.choice(templates).format(A=event_point_1, B=event_point_2)

    def __str__(self):
        if self.relationship_type() == 'simultaneous':
            return self.get_simultaneous_sentence(self._event_point_1, self._event_point_2)
        elif self.relationship_type() == 'before':
            return self.get_before_sentence(self._event_point_1, self._event_point_2)
        elif self.relationship_type() == 'after':
            return self.get_after_sentence(self._event_point_1, self._event_point_2)
