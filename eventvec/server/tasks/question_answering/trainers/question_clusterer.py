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
        "supposed to"
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

cluster_keys = {k: set(v) for k, v in cluster_keys.items()}

class QuestionClusterer:
    def cluster(self, question):
        used = False
        cluster_map = [0 for _ in range(self.num_clusters())]
        for cluster_key_i, (cluster_key, cluster_key_values) in enumerate(cluster_keys.items()):
            for cluster_key_value in cluster_key_values:
                if cluster_key_value in question and used is False:
                    cluster_map[cluster_key_i] = 1
                    used = True
        if used is False:
            cluster_map[-1] = 1
        return cluster_map
        
    def num_clusters(self):
        return len(cluster_keys.keys()) + 1