from collections import defaultdict
import json

cluster_keys = {
    "What events have already finished?":[
        "What events have already finished?"
    ],
    "What events have begun but has not finished?": [
        "What events have begun but has not finished?"
    ],
    "What might happen during":[
        "What might happen during",
        "might",
        "may"
    ],
    "not":{
        "not",
        "didn't"
    },
    "What will happen":{
        "will",
    },
    "What began":{
        "began",
        "started",
    },
    "finished":{
        "finished",
        "ended",
    },
        "What happened after": [
        "happened after",
        "What events happened after",
        "do after",
        "What event happened after",
        "occurred after",
        "happen after",
        "happens after",
        "happening after",
        "took place after",
        "take place after"
        "immediately after",
        "happened right after",
    ],
    "What happened before":[
        "happened before",
        "happened right before",
        "What events happened before",
        "do before",
        "What event happened before"
        "occurred before",
        "happen before",
        "happens before",
        "happening before",
        "took place before",
        "take place before"
        "immediately before",
        "happened before",
        "happened right before",
    ],
    "What happened during":[
        "happened during",
        "happen during",
        "do during",
        "What happened while",
        "What event happened during",
        "What events happened during",
        "What is happening during",
        "What is happening while",
        "occurred during",
        "happens during",
        "happening during",
        "happened while",
        "happen while",
        "occurred while",
        "happens while",
        "happening while",
        "happened when",
        "happen when",
        "occurred when",
        "happens when",
        "happening when",
        "took place when",
        "take place when"
    ],
    "happened since": [
        "happened since"
    ]
}

def main():
    clusters = defaultdict(set)
    questions = []
    remaining = []
    with open('eventvec/server/data/torque/extra/dev_questions.txt') as f:
        for line in f:
            added = False
            for cluster_key, cluster_key_values in cluster_keys.items():
                for cluster_key_value in cluster_key_values:
                    if cluster_key_value in line and added is False:
                        clusters[cluster_key].add(line.strip('\n'))
                        added = True
            if added is False:
                remaining.append(line.strip('\n'))
    print(remaining)
    print(len(remaining))
    for key in clusters:
        print(key, len(clusters[key]))
    print(clusters["What happened during"])
    with open("eventvec/server/data/torque/extra/clusters.json", "wt") as f:
        json.dump({k: list(v) for k, v in clusters.items()}, f, indent=4)

if __name__ == '__main__':
    main()