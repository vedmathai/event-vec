import json

def filterf(element):
    return all(i == 'entailment' for i in element['annotator_labels'])

with open('/home/lalady6977/oerc/projects/data/multinli_1.0/multinli_1.0/multinli_1.0_dev_matched.jsonl') as f:
    data = []
    for r in f:
        data.append(json.loads(r))
    data = filter(filterf, data)
    for i in data:
        print()
        print(i['sentence1'], i['sentence2'])