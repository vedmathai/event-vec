from explore.reader import epub2text
import spacy
from explore.dep_parser import get_path
import json
from collections import defaultdict
import os
import re

nlp = spacy.load("en_core_web_sm")


preps_file = 'relationships.json'
with open(preps_file) as f:
    preps = json.load(f)

folder = '/Users/vedmathai/Documents/python/FeverousWikiv1'

def fix_sentence(sentence):
    sentence = re.sub('(\[\[[^\|]*\||\]\])',  '', sentence)
    return sentence


contents = []
for file_name in os.listdir(folder)[0:5]:
    file_path = os.path.join(folder, file_name)
    with open(file_path, encoding='utf-8') as f:
        try:
            for line in f:
                sentence_content = []
                content = json.loads(line)
                for key in content:
                    if key[0:8] == 'sentence':
                        sentence_content += [fix_sentence(content[key])]
                contents += [' '.join(sentence_content)]
        except UnicodeDecodeError:
            continue

chapter_count = defaultdict(lambda: set())
chapter_total_sents = defaultdict(lambda: set())
#contents = ["in 1927 she published a children's book, the turned about girls, along with a risqué drama novel called their own desire."]
for chapteri, chapter in enumerate(contents):
    spacy_doc = nlp(chapter)
    chapter_total_sents[chapteri] = len(list(spacy_doc.sents))
    used = False
    for sentence_i, sentence in enumerate(spacy_doc.sents):
        for token_1i, token_1 in enumerate(sentence):
            for token_2i, token_2 in enumerate(sentence):
                if token_1i == token_2i:
                    continue
                if not (token_1.pos_ in ['VERB'] and (token_2.pos_ in ['VERB'] or token_2.ent_type_ in ['DATE'])):
                    continue
                dep = get_path(spacy_doc, token_1.i, token_2.i)
                dep_tup = []
                for ii, i in enumerate(dep):
                    if i.dep_ == 'prep':
                        prep_i = ii
                        dep_tup += [i.lemma_]
                    else:
                        #dep_tup += [i.pos_]
                        pass
                dep_tup = '|'.join(dep_tup)
                if len(dep_tup) > 0:
                    if len(dep) > 0:
                        for i in preps:
                            if i == dep_tup:
                                #print(dep)
                                #print(sentence)
                                #print('-----')
                                #print('\n' * 4)
                                used = True
                                chapter_count[chapteri].add((sentence_i, chapteri))
        if used is False:
            print(sentence)
    
                
    print(chapteri, f'{len(chapter_count[chapteri])}\{chapter_total_sents[chapteri]}')
print({k: f'{float(len(v))}/{chapter_total_sents[k]}' for k, v in chapter_count.items()})
