before_temporal_train_templates = [
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

after_temporal_train_templates = [
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

simultaneous_temporal_train_templates = [
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

simultaneous_temporal_test_templates = [
    "The {A} happened simultaneous to the {B}",
]

after_temporal_test_templates = [
    "The {A} happened after the {B}",
]

before_temporal_test_templates = [
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

templates_dict = {
    'temporal':{
        'train': {
            'before': before_temporal_train_templates,
            'after': after_temporal_train_templates,
            'simultaneous': simultaneous_temporal_train_templates,
        },
        'test': {
            'before': before_temporal_test_templates,
            'after': after_temporal_test_templates,
            'simultaneous': simultaneous_temporal_test_templates,
        }
    },
    'spatial':{
        'train': {
            'before': before_spatial_train_templates,
            'after': after_spatial_train_templates,
            'simultaneous': simultaneous_spatial_train_templates,
        },
        'test': {
            'before': before_spatial_test_templates,
            'after': after_spatial_test_templates,
            'simultaneous': simultaneous_spatial_test_templates,
        }
    },
    'logical':{
        'train': {
            'before': before_logical_train_templates,
            'after': after_logical_train_templates,
            'simultaneous': simultaneous_logical_train_templates,
        },
        'test': {
            'before': before_logical_train_templates,
            'after': after_logical_train_templates,
            'simultaneous': simultaneous_logical_train_templates,
        }
    }
}