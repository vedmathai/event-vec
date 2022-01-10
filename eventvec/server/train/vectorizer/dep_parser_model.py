import spacy
import re
from collections import defaultdict
from collections import deque
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a sentence.")


def main():
    while True:
        sent = input('\n\n\nEnter sentence:\n\n')
        sent.replace('\n', ' ')
        sent = re.sub("\[[0-9]*\]", "", sent) 
        doc = nlp(sent)
        for sentence in doc.sents:
            print()
            print(sentence)
            Node.clear()
            root = sentence.root
            traverse(root)
            root = Node.root
            for node in Node.nodes:
                subj = follow_down_orth(node, ['nsubj', 'nsubjpass'])
                obj = follow_down_orth(node, ['prep>pobj', 'dobj', 'pobj', 'pobj>compound', 'xcomp', 'acomp>prep>pobj'])
                if has_item(subj) and has_item(obj):
                    print(subj, node.orth, obj)
            """
                    if key in ['nsubj', 'nsubjpass']:
                        obj = []
                        subj = [k.orth for k in node.down[key]]
                        for obj_key in ['pobj', 'dobj']:
                            if obj_key in node.down:
                                obj += [k.orth for k in node.down[obj_key]]
                        print(subj, node.orth, obj)
                #if row[0] in ['compound']:
                #    print('-'*4, '>', row)
            """


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
        return [w.orth for w in fd]


def follow_down(node, paths):
    words = []
    paths = [k.split('>') for k in paths]
    enumerated_paths = enumerate_paths(node)
    for path in paths:
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


class Node():
    nodes_dict = {}
    root = None
    nodes = set()

    def __init__(self, idx, orth, is_root=False):
        self.up = defaultdict(lambda: [])
        self.down = defaultdict(lambda: [])
        self.idx = idx
        self.orth = orth
        self.is_root = is_root
        if is_root is True:
            Node.root = self
        Node.nodes.add(self)

    @staticmethod
    def add_r(sp_child_node, sp_head_node):
        dep = sp_child_node.dep_
        if sp_head_node not in Node.nodes_dict:
            is_root = False
            if sp_head_node.dep_ == 'ROOT':
                is_root = True
            Node.nodes_dict[sp_head_node] = Node(sp_head_node.idx, sp_head_node.orth_, is_root)
        if sp_child_node not in Node.nodes_dict:
            Node.nodes_dict[sp_child_node] = Node(sp_child_node.idx, sp_child_node.orth_)
        child = Node.nodes_dict[sp_child_node]
        parent = Node.nodes_dict[sp_head_node]
        parent.down[dep].append(child)
        child.up[dep].append(parent)


    @staticmethod
    def clear():
        Node.nodes_dict = {}
        Node.root = None
        Node.nodes = set()

if __name__ == '__main__':
    main()