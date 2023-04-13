import json

opposite_dict = {}
opposite_pairs = [
    ("INCLUDES", "IS_INCLUDED"),
    ("AFTER", "BEFORE"),
    ("IBEFORE", "IAFTER"),
    ("ENDS", "BEGINS"),
    ("IDENTITY", "IDENTITY"),
    ("SIMULTANEOUS", "SIMULTANEOUS"),
    ("CULMINATES", "INITIATES"),
    ("ENDED_BY", "BEGUN_BY"),
    ("INITIATES", "TERMINATES"),
    ("DURING", "DURING"),
]

for pair in opposite_pairs:
    opposite_dict[pair[0]] = pair[1]
    opposite_dict[pair[1]] = pair[0]

prepositions_file = "eventvec/server/data/timebank_prepositions.json"
with open(prepositions_file) as f:
    prep_to_relationships = json.load(f)

prep_to_opposite_relationships = {}
for prep in prep_to_relationships:
    prep_to_opposite_relationships[prep] = {}
    for rel in prep_to_relationships[prep]:
        opposite_rel = opposite_dict[rel]
        prep_to_opposite_relationships[prep][opposite_rel] = prep_to_relationships[prep][rel]
