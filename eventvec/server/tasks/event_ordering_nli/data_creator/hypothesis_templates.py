hypothesis_templates = {
    'temporal': {
        'before': [
            '{} happens before {}'
        ],
        'after': [
            '{} happens after {}'
        ],
        'overlaps': [
            '{} overlaps with {}'
        ]
    },
    'spatial': {
        'before': [
            '{} is completely west of {}'
        ],
        'after': [
            '{} is completely east of {}'
        ],
        'overlaps': [
            '{} shares at least one longitude with {}',
        ]
    },
    'logical': {
        'before': [
            '{} < {}'
        ],
        'after': [
            '{} > {}'
        ],
        'overlaps': [
            '{} = {}'
        ]
    }
}
