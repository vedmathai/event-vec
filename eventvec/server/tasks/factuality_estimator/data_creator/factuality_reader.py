import json
import csv

def filterf(element):
    judgements = element['results']['judgments']
    scores = [int(k['data']['belief_question']) for k in judgements]
    return any(i<0 for i in scores)

def write(negatives):
    with open('/home/lalady6977/oerc/projects/data/factuality_experiments/data/TBAQ/crowdflower/negatives_2.csv', 'wt') as f:
        writer = csv.writer(f, delimiter=',')
        negatives = sorted(negatives, key=lambda x: len(x[0]))
        for negative in negatives:
            writer.writerow(negative)

with open('/home/lalady6977/oerc/projects/data/factuality_experiments/data/TBAQ/crowdflower/tempeval.belief.2.json') as f:
    data = []
    set_of_data = set()
    for r in f:
        data.append(json.loads(r))
    data = filter(filterf, data)
    seen = set()
    negatives = []
    for ii, i in enumerate(data):
        sentence = ' '.join(i["data"]["tokens"])
        if (sentence, i["data"]["event_string"]) in seen:
            continue
        seen.add((sentence, i["data"]["event_string"]))
        print(ii, sentence)
        negatives.append([sentence, i["data"]["event_string"], i['results']['belief_question']['agg']])
    write(negatives)
