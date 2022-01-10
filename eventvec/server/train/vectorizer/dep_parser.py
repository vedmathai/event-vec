import spacy

nlp = spacy.load("en_core_web_sm")


def get_path(doc, i, j):
    if i <= j:
        left = i
        right = j
    if i > j:
        left = j
        right = i
    if len(doc) <= right:
        return []
    for token in doc:
        if token.i == left:
            left_token = token
        if token.i == right:
            right_token = token
    left_path_to_root = []
    right_path_to_root = []
    left_pointer = left_token
    while True:
        left_path_to_root += [left_pointer]
        if left_pointer.dep_ == 'ROOT':
            break
        left_pointer = left_pointer.head
    right_pointer = right_token
    while True:
        right_path_to_root += [right_pointer]
        if right_pointer.dep_ == 'ROOT':
            break
        right_pointer = right_pointer.head
    min_len = min(len(left_path_to_root), len(right_path_to_root))
    left_path_to_root = left_path_to_root[::-1]
    right_path_to_root = right_path_to_root[::-1]
    non_breaker = 0
    i = 0
    while True:
        if i < min_len:
            if left_path_to_root[i] != right_path_to_root[i]:
                non_breaker = i-1
                break
        elif i == min_len:
            non_breaker = i-1
            break
        i += 1
    left = left_path_to_root[non_breaker+1:]
    mid = [left_path_to_root[non_breaker]]
    right = right_path_to_root[non_breaker+1:]
    seq = left[::-1] + mid + right
    if i > j:
        seq = seq[::-1]
    return seq


if __name__=='__main__':
    doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
    get_path(doc, 0, 5)