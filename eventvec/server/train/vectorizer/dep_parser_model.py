import spacy
import re
from collections import defaultdict
from collections import deque


from eventvec.utils.spacy_utils.utils import get_spacy_doc

nlp = spacy.load("en_core_web_trf")


class Node():
    nodes_dict = {}
    root = None
    nodes = set()

    def __init__(self, spacy_token, is_root=False):
        self.up = defaultdict(lambda: [])
        self.down = defaultdict(lambda: [])
        self.spacy_token = spacy_token
        self._is_root = is_root
        if is_root is True:
            Node.root = self
        Node.nodes.add(self)

    @staticmethod
    def add_r(spacy_child_node, spacy_head_node):
        dep = spacy_child_node.dep_
        if spacy_head_node not in Node.nodes_dict:
            is_root = False
            if spacy_head_node.dep_ == 'ROOT':
                is_root = True
            Node.nodes_dict[spacy_head_node] = Node(spacy_head_node, is_root)
        if spacy_child_node not in Node.nodes_dict:
            Node.nodes_dict[spacy_child_node] = Node(spacy_child_node)
        child = Node.nodes_dict[spacy_child_node]
        parent = Node.nodes_dict[spacy_head_node]
        parent.down[dep].append(child)
        child.up[dep].append(parent)

    def head(self):
        return Node.nodes_dict[self.spacy_token.head]

    def lemma(self):
        return self.spacy_token.lemma_

    def i(self):
        return self.spacy_token.i

    def orth(self):
        return self.spacy_token.orth_

    def dep(self):
        return self.spacy_token.dep_

    def pos(self):
        return self.spacy_token.pos_

    def is_root(self):
        return self._is_root

    @staticmethod
    def clear():
        Node.nodes_dict = {}
        Node.root = None
        Node.nodes = set()

def parse_sentence(sentence):
    root = sentence.root
    traverse(root)
    root = Node.root
    psentence = []
    for token in sentence:
        psentence += [Node.nodes_dict[token]]
    return root, psentence

def has_item(item):
    if isinstance(item, list) and len(item) > 0:
        return True
    if item is None:
        return False
    return False


def dedupe_paths(paths):
    paths = set(tuple(k) for k in paths)
    paths = [list(k) for k in paths]
    return paths


def enumerate_paths(node):
    paths = enumerate_paths_aux(node)
    new_paths = []
    for path in paths:
        for i in range(1, len(path)+1):
            new_paths += [path[0:i]]
    new_paths = dedupe_paths(new_paths)
    return new_paths


def enumerate_paths_aux(node):
    paths = []
    for cdep, children in node.down.items():
        if cdep == 'ROOT':
            continue
        for child in children:
            child_paths = enumerate_paths_aux(child)
            for path in child_paths:
                paths += [[cdep] + path]
    if len(node.down) == 0:
        paths = [[]]
    paths = dedupe_paths(paths)
    return paths


def follow_down_orth(node, paths):
    fd = follow_down(node, paths)
    if fd is not None:
        return [w.orth() for w in fd]


def follow_down(node, paths):
    words = []
    paths = [k.split('>') for k in paths]
    enumerated_paths = enumerate_paths(node)
    for path in paths:
        if path == ['ROOT']:
            words += [node]
        if path in enumerated_paths:
            curr_nodes = [node]
            for elementi, element in enumerate(path):
                new_nodes = []
                for temp_node in curr_nodes:
                    new_nodes.extend(temp_node.down[element])
                curr_nodes = new_nodes
                if elementi == len(path)-1:
                    words += curr_nodes
    return words


def traverse(root):
    traversal = []
    q = deque([root])
    while len(q) > 0:
        node = q.popleft()
        Node.add_r(node, node.head)
        q.extend(node.lefts)
        q.extend(node.rights)
    return traversal


def get_path(sentence, i, j):
    if i <= j:
        left = i
        right = j
    if i > j:
        left = j
        right = i
    if sentence[-1].i() < right:
        return []
    for token in sentence:
        if token.i() == left:
            left_token = token
        if token.i() == right:
            right_token = token
    left_path_to_root = []
    right_path_to_root = []
    left_pointer = left_token
    right_pointer = right_token
    while True:
        left_path_to_root += [left_pointer]
        if left_pointer.is_root():
            break
        left_pointer = left_pointer.head()
    while True:
        right_path_to_root += [right_pointer]
        if right_pointer.is_root():
            break
        right_pointer = right_pointer.head()
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

#Notes for later
#subj = follow_down_orth(node, ['nsubj', 'nsubjpass'])
#obj = follow_down_orth(node, ['prep>pobj', 'dobj', 'pobj', 'pobj>compound', 'xcomp', 'acomp>prep>pobj'])
