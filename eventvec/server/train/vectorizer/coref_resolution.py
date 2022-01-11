import coreferee, spacy
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')
doc = nlp('Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.')
doc._.coref_chains.print()
# Output:
#
# 0: he(1), his(6), Peter(9), He(16), his(18)
# 1: work(7), it(14)
# 2: [He(16); wife(19)], they(21), They(26), they(31)
# 3: Spain(29), country(34)
#
print(doc._.coref_chains.resolve(doc[31]))
# Output: