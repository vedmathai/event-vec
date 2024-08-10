import csv
from jadelogs import JadeLogger

jade_logger = JadeLogger()
filepath = jade_logger.file_manager.data_filepath('exaggeration.csv')


cases = [
    'modal',
    'speech_verb',
    'verb',
'noun',
    'temporal',
    'connector',
    'scalar_adjective',
    'quantifier',
    'number',
    'anaphora',
]

factor2 = [
    'first',
    'second',
]

factor3 = [
    'agree',
    'disagree',
]

with open('exaggeratiosn.csv', 'wt') as f:
    writer = csv.writer(f)
    writer.writerow(['sr', 'case', 'factor2', 'factor3', 'speaker A', 'speaker B'])
    serial = 1
    for case in cases:
        for f2 in factor2:
            for f3 in factor3:
                for f4 in range(1, 11):
                    writer.writerow([serial, case, f2, f3, ''])
                    serial += 1