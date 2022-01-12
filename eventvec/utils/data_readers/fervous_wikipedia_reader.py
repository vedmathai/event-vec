
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