import json


prepositions_file = "eventvec/server/data/timebank_prepositions.json"
with open(prepositions_file) as f:
    prep_to_relationships = json.load(f)
