import re

from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.common.lists.said_verbs import contrasting_conjunctions

complement_deps = ['ccomp', 'xcomp']


class ContrastFeaturizer:
    def __init__(self):
        self._linguistic_featurizer = LinguisticFeaturizer()
        self._fns = [
            #self._despite,
            self._but,
            self._however,
            self._though,
            self._in_spite_of,
            self._regardless,
        ]
        self._fns2 = [

            self._on_the_other_hand,
            self._nevertheless,
            self._instead_of,
            self._rather_than,
            self._in_fact,
            self._in_reality,
            self._in_contrast,
            self._in_comparison,
            self._on_the_contrary,
        ]

    def _despite(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for tokeni, token in enumerate(sentence.tokens()):
                if token.text() == 'despite':
                    sorted_children, sorted_parents = self.get_parents_and_children(token)
                    para += ' ' + '{} CONTRASTS {}'.format(sorted_parents, sorted_children)
        return para


    def _but(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token in sentence.tokens():
                if token.text() in ['but']:
                    root = token.parent()
                    children = []
                    if root is None:
                        continue
                    if 'conj' in root.children():
                        child = root.children()['conj'][0]
                        #children = [child] + children.all_children() # delete this
                        children = [child] + child.traverse_all_children(complement_deps)
                        children = [child] + child.children().get('nsubj', []) 
                        children = children + child.children().get('dobj', [])
                        children = children + child.children().get('attr', [])


                    #parents = root.traverse_all_children(['conj', 'cc'] + complement_deps)
                    parents = root.children().get('nsubj', [])
                    parents = root.children().get('dobj', [])
                    parents = root.children().get('attr', [])


                    parents = parents + [root]
                    #parents = [root] + root.all_children() # delete this
                    sorted_children = sorted(children, key=lambda x: x.i())
                    sorted_parents = sorted([i for i in parents if i not in children + [token]], key=lambda x: x.i())
                    sorted_children_text = ' '.join([i.text() for i in sorted_children]).strip().strip('.')
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    if token.i() < sorted_parents[0].i():
                        para += ' ' + 'CONTRAST_START {} CONTRASTS {} CONTRAST_STOP'.format(sorted_parents_text, '')
                    else:
                        para += ' ' + 'CONTRAST_START {} CONTRASTS {} CONTRAST_STOP'.format(sorted_children_text, sorted_parents_text)
        return para


    def _however(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token in sentence.tokens():
                if token.text() in ['however']:
                    root = token.parent()
                    children = []
                    if 'ccomp' in root.children():
                        child = root.children()['ccomp'][0]
                        #children = [children] + children.all_children() # delete
                        #children = [children] + children.traverse_all_children(complement_deps)
                        children = [child] + child.children().get('nsubj', []) 
                        children = children + child.children().get('dobj', [])
                        children = children + child.children().get('attr', [])

                    #parents = root.traverse_all_children(complement_deps)
                    #parents = parents + [root]
                    #parents = [root] + root.all_children()
                    parents = root.children().get('nsubj', [])
                    parents = root.children().get('dobj', [])
                    parents = root.children().get('attr', [])
                    parents = parents + [root]
                    sorted_children = sorted(children, key=lambda x: x.i())
                    sorted_parents = sorted([i for i in parents if i not in children + [token]], key=lambda x: x.i())
                    sorted_children_text = ' '.join([i.text() for i in sorted_children]).strip().strip('.')
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    if token.i() < sorted_parents[0].i():
                        para += ' ' + '{} CONTRASTS {}'.format(sorted_parents_text, '')
                    else:
                        para += ' ' + '{} CONTRASTS {}'.format(sorted_parents_text, sorted_children_text)
        return para


    def _though(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token in sentence.tokens():
                if token.text() in ['though']:
                    root = token.parent()
                    parents = []
                    if root.dep() == 'advcl':
                        parent = root.parent()
                        #parents = [parent] + parent.all_children()
                        #parents = [parent] + parent.traverse_all_children(complement_deps + ['advcl'])
                        parents = root.children().get('nsubj', [])
                        parents = root.children().get('dobj', [])
                        parents = root.children().get('attr', [])

                        parents = parents + [root]
                    #children = root.traverse_all_children(complement_deps)
                    #children = children + [root]
                    child = root
                    children = [child] + child.children().get('nsubj', []) 
                    children = children + child.children().get('dobj', [])
                    children = children + child.children().get('attr', [])

                    #children = [root] + root.all_children()
                    sorted_children = sorted([i for i in children if i not in [token]], key=lambda x: x.i())
                    sorted_parents = sorted([i for i in parents if i not in children + [token]], key=lambda x: x.i())
                    sorted_children_text = ' '.join([i.text() for i in sorted_children]).strip().strip('.')
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    para += ' ' + '{} CONTRASTS {}'.format(sorted_children_text, sorted_parents_text)
        return para


    def _in_spite_of(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token_i, token in enumerate(sentence.tokens()):
                if self.match(token_i, sentence.tokens(), 'in spite of'):
                    in_spite_of = [token]
                    parent = token.parent()
                    #parents = [parent] + parent.traverse_all_children(complement_deps + ['prep'])
                    parents = parent.children().get('nsubj', [])
                    parents = parent.children().get('dobj', []) 
                    parents = parent.children().get('attr', []) 

                    parents = parents + [parent]
                    #children = token.all_children()[0]
                    children = [token] + token.children().get('nsubj', []) 
                    children = children + token.children().get('dobj', [])
                    children = children + token.children().get('attr', [])

                    in_spite_of += [children]
                    children = token.all_children()[0]
                    in_spite_of += [children]
                    #children = children.traverse_all_children(complement_deps)
                    sorted_children = sorted([i for i in children if i not in in_spite_of], key=lambda x: x.i())
                    sorted_parents = sorted([i for i in parents if i not in children + in_spite_of], key=lambda x: x.i())
                    sorted_children_text = ' '.join([i.text() for i in sorted_children]).strip().strip('.')
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    para += ' ' + '{} CONTRASTS {}'.format(sorted_parents_text, sorted_children_text)
        return para



    def _regardless(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token in sentence.tokens():
                if token.text() in ['regardless']:
                    root = token.parent()
                    parent = root
                    #parents = [root] + root.traverse_all_children(complement_deps + ['advmod'])
                    parents = parent.children().get('nsubj', [])
                    parents = parent.children().get('dobj', [])
                    parents = parent.children().get('attr', [])

                    parents = parents + [parent]
                    children = [token] + token.children().get('nsubj', []) 
                    children = children + token.children().get('dobj', [])
                    children = children + token.children().get('attr', [])
                
                    #children = token.traverse_all_children(complement_deps)
                    sorted_children = sorted([i for i in children if i not in [token]], key=lambda x: x.i())
                    sorted_parents = sorted([i for i in parents if i not in children + [token]], key=lambda x: x.i())
                    sorted_children_text = ' '.join([i.text() for i in sorted_children]).strip().strip('.')
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    para += ' ' + '{} CONTRASTS {}'.format(sorted_children_text, sorted_parents_text)
        return para


    def _on_the_other_hand(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token_i, token in enumerate(sentence.tokens()):
                if self.match(token_i, sentence.tokens(), 'on the other hand'):
                    root = token.parent()
                    parents = [root] + root.traverse_all_children(complement_deps)
                    children = token.traverse_all_children(complement_deps) + [token]
                    sorted_parents = sorted([i for i in parents if i not in children + [token]], key=lambda x: x.i())
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    para += ' ' + '{} CONTRASTS'.format(sorted_parents_text)
        return para


    def _nevertheless(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token_i, token in enumerate(sentence.tokens()):
                if self.match(token_i, sentence.tokens(), 'nevertheless'):
                    root = token.parent()
                    parents = [root] + root.traverse_all_children(complement_deps)
                    children = token.traverse_all_children(complement_deps) + [token]
                    sorted_parents = sorted([i for i in parents if i not in children + [token]], key=lambda x: x.i())
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    para += ' ' + '{} CONTRASTS'.format(sorted_parents_text)
        return para


    def _instead_of(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token_i, token in enumerate(sentence.tokens()):
                if self.match(token_i, sentence.tokens(), 'instead of'):
                    instead_of = [token]
                    of = token.parent()
                    instead_of += [of]
                    parent = of.parent()
                    parents = [parent] + parent.traverse_all_children(complement_deps)
                    children = of.traverse_all_children(complement_deps)
                    sorted_children = sorted([i for i in children if i not in instead_of], key=lambda x: x.i())
                    sorted_parents = sorted([i for i in parents if i not in children + instead_of], key=lambda x: x.i())
                    sorted_children_text = ' '.join([i.text() for i in sorted_children]).strip().strip('.')
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    para += ' ' + '{} CONTRASTS {}'.format(sorted_parents_text, sorted_children_text)
        return para


    def _rather_than(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token_i, token in enumerate(sentence.tokens()):
                if self.match(token_i, sentence.tokens(), 'rather than'):
                    rather_than = [token]
                    than = token.parent()
                    rather_than += [than]
                    parent = than.parent()
                    parents = [parent] + parent.traverse_all_children(complement_deps)
                    children = than.traverse_all_children(complement_deps)
                    sorted_children = sorted([i for i in children if i not in rather_than], key=lambda x: x.i())
                    sorted_parents = sorted([i for i in parents if i not in children + rather_than], key=lambda x: x.i())
                    sorted_children_text = ' '.join([i.text() for i in sorted_children]).strip().strip('.')
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    para += ' ' + '{} CONTRASTS {}'.format(sorted_parents_text, sorted_children_text)
        return para


    def _in_fact(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token_i, token in enumerate(sentence.tokens()):
                if self.match(token_i, sentence.tokens(), 'in fact'):
                    root = token.parent()
                    parents = [root] + root.traverse_all_children(complement_deps)
                    children = token.traverse_all_children(complement_deps) + [token]
                    sorted_parents = sorted([i for i in parents if i not in children + [token]], key=lambda x: x.i())
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    para += ' ' + '{} CONTRASTS'.format(sorted_parents_text)
        return para


    def _in_reality(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token_i, token in enumerate(sentence.tokens()):
                if self.match(token_i, sentence.tokens(), 'in reality'):
                    in_reality = [token] + [sentence.tokens()[token_i + 1]]
                    root = token.parent()
                    children = []
                    if 'advcl' in root.children():
                        children = root.children()['advcl'][0]
                        children = [children] + children.traverse_all_children(complement_deps)
                    parents = root.traverse_all_children(complement_deps)
                    parents = parents + [root]
                    sorted_children = sorted([i for i in children if i not in in_reality], key=lambda x: x.i())
                    sorted_parents = sorted([i for i in parents if i not in children + [token]], key=lambda x: x.i())
                    sorted_children_text = ' '.join([i.text() for i in sorted_children]).strip().strip('.')
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    if token.i() < sorted_parents[0].i():
                        para += ' ' + '{} CONTRASTS {}'.format(sorted_parents_text, '')
                    else:
                        para += ' ' + '{} CONTRASTS {}'.format(sorted_parents_text, sorted_children_text)
        return para


    def _in_contrast(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token_i, token in enumerate(sentence.tokens()):
                if self.match(token_i, sentence.tokens(), 'in contrast'):
                    in_contrast = [token]
                    contrast = sentence.tokens()[token_i + 1]
                    in_contrast += [contrast]
                    parent = token.parent()
                    parents = [parent] + parent.traverse_all_children(complement_deps)
                    children = contrast.traverse_all_children(complement_deps)
                    sorted_children = sorted([i for i in children if i not in in_contrast], key=lambda x: x.i())
                    sorted_parents = sorted([i for i in parents if i not in children + in_contrast], key=lambda x: x.i())
                    sorted_children_text = ' '.join([i.text() for i in sorted_children]).strip().strip('.')
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    para += ' ' + '{} CONTRASTS {}'.format(sorted_parents_text, sorted_children_text)
        return para

    def _in_comparison(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token_i, token in enumerate(sentence.tokens()):
                if self.match(token_i, sentence.tokens(), 'in comparison'):
                    in_comparison_to = [token] + [sentence.tokens()[token_i + 1]]
                    comparison = sentence.tokens()[token_i + 1]
                    parent = token.parent()
                    parents = [parent] + parent.traverse_all_children(complement_deps)
                    children = comparison.traverse_all_children(complement_deps)
                    sorted_children = sorted([i for i in children if i not in in_comparison_to], key=lambda x: x.i())
                    sorted_parents = sorted([i for i in parents if i not in children + in_comparison_to], key=lambda x: x.i())
                    sorted_children_text = ' '.join([i.text() for i in sorted_children]).strip().strip('.')
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    para += ' ' + '{} CONTRASTS {}'.format(sorted_parents_text, sorted_children_text)
        return para


    def _on_the_contrary(self, fdoc):
        para = ''
        for sentence in fdoc.sentences():
            for token_i, token in enumerate(sentence.tokens()):
                if self.match(token_i, sentence.tokens(), 'on the contrary'):
                    root = token.parent()
                    parents = [root] + root.traverse_all_children(complement_deps)
                    children = token.traverse_all_children(complement_deps) + [token]
                    sorted_parents = sorted([i for i in parents if i not in children + [token]], key=lambda x: x.i())
                    sorted_parents_text = ' '.join([i.text() for i in sorted_parents]).strip().strip('.')
                    para += ' ' + '{} CONTRASTS'.format(sorted_parents_text)
        return para

    def match(self, idx, tokens, phrase):
        phrase_split = phrase.split()
        phrase_len = len(phrase_split)
        token_texts = [i.text() for i in tokens[idx: idx + phrase_len]]
        if token_texts == phrase_split:
            return True
        return False

    def get_parents(self, token):
        parent = token
        while parent.dep() is not complement_deps:
            if parent.parent() is not None:
                parent = parent.parent()
            else:
                break
        parents = parent.traverse_all_children(complement_deps)
        return parents

    def get_children(self, token):
        children = token.traverse_all_children([])
        return children
    
    def get_parents_and_children(self, token):
        children = self.get_children(token)
        parents = self.get_parents(token)
        sorted_children = sorted(children, key=lambda x: x.i())
        sorted_parents = sorted([i for i in parents if i not in children + [token]], key=lambda x: x.i())
        sorted_children = ' '.join([i.text() for i in sorted_children])
        sorted_parents = ' '.join([i.text() for i in sorted_parents])
        return sorted_children, sorted_parents

    def featurize(self, sentence):
        fdoc = self._linguistic_featurizer.featurize_document(sentence.lower())
        para = ''
        for fn in self._fns:
            para += ' ' + fn(fdoc)
        para = re.sub('\.|,', '', para)
        #if 'CONTRASTS' in para:
        #    return 'CONTRASTS' + ' ' + sentence
        #else:
        #    return para + ' ' + sentence
        return para + ' ' + sentence


if __name__ == '__main__':
    fc = ContrastFeaturizer()
    t1 = fc.featurize("On the contrary, John is very fond of you.")
    print(t1)
