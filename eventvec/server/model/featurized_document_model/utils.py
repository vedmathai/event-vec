
def traverse_up(current_token):
    path = []
    while current_token.dep() != 'ROOT':
        path.append(current_token)
        current_token = current_token.parent()
    path.append(current_token)
    return path[::-1]


def find_common_parent(path_1, path_2):
    smaller = path_1
    larger = path_2

    if len(path_1) > len(path_2):
        smaller = path_2
        larger = path_1
    i = 0
    if larger[:len(smaller)] == smaller:
        return len(smaller) - 1
    while i != len(smaller) - 1:
        s_token = smaller[i+1]
        l_token = larger[i+1]
        if (s_token.dep(), s_token.i()) != (l_token.dep(), l_token.i()):
            common = i
            break
        i += 1
    return common

def resolve_coreference_pointers(document):
    i2token = {}
    for sentence in document.sentences():
        for token in sentence.tokens():
            i2token[token.i()] = token
    for sentence in document.sentences():
        for token in sentence.tokens():
            if token.coreference() is not None:
                coreference = [i2token[i] for i in token.coreference()]
                token.set_coreference(coreference)
    return document
