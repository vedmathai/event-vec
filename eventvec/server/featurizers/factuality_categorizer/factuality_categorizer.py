from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.common.lists.said_verbs import future_modals, confident_said_verbs, believe_verbs, expect_verbs, modal_adverbs, modal_adjectives, negation_words, past_perf_aux, pres_perf_aux

complement_deps = ['ccomp', 'xcomp']

class FeaturesArray:
    def __init__(self):
        self._is_negated = False
        self._has_modal = False
        self._has_modal_adverb = False
        self._has_modal_adjective = False
        self._is_subordinate_of_said = False
        self._is_subordinate_of_believe = False
        self._is_subordinate_of_if = False
        self._is_speech_act = False
        self._is_belief_act = False
        self._is_subordinate_of_expects = False
        self._is_subordinate_of_then = False
        self._has_negation_words = False
        self._is_past_tense = False
        self._is_present_tense = False
        self._is_future_tense = False
        self._is_continuous_aspect = False
        self._is_perfect_aspect = False

    def is_negated(self):
        return self._is_negated
    
    def set_is_negated(self, is_negated):
        self._is_negated = is_negated

    def set_is_subordinate_of_said(self, is_subordinate_of_said):
        self._is_subordinate_of_said = is_subordinate_of_said

    def set_is_subordinate_of_believe(self, is_subordinate_of_believe):
        self._is_subordinate_of_believe = is_subordinate_of_believe

    def set_is_subordinate_of_if(self, is_subordinate_of_if):
        self._is_subordinate_of_if = is_subordinate_of_if

    def set_is_speech_act(self, is_speech_act):
        self._is_speech_act = is_speech_act

    def set_is_subordinate_of_expects(self, is_subordinate_of_expects):
        self._is_subordinate_of_expects = is_subordinate_of_expects

    def set_is_subordinate_of_then(self, is_subordinate_of_then):
        self._is_subordinate_of_then = is_subordinate_of_then

    def set_has_modal_adverb(self, has_modal_adverb):
        self._has_modal_adverb = has_modal_adverb

    def set_has_modal_adjective(self, has_modal_adjective):
        self._has_modal_adjective = has_modal_adjective

    def set_has_negation_words(self, has_negation_words):
        self._has_negation_words = has_negation_words

    def set_is_perfect_aspect(self, is_perfect_aspect):
        self._is_perfect_aspect = is_perfect_aspect

    def set_is_continuous_aspect(self, is_continuous_aspect):
        self._is_continuous_aspect = is_continuous_aspect

    def set_is_past_tense(self, is_past_tense):
        self._is_past_tense = is_past_tense

    def set_is_present_tense(self, is_present_tense):
        self._is_present_tense = is_present_tense

    def is_subordinate_of_expects(self):
        return self._is_subordinate_of_expects

    def is_speech_act(self):
        return self._is_speech_act

    def is_subordinate_of_then(self):
        return self._is_subordinate_of_then
    
    def set_is_belief_act(self, is_belief_act):
        self._is_belief_act = is_belief_act

    def is_belief_act(self):
        return self._is_belief_act
    
    def set_is_belief_act(self, is_belief_act):
        self._is_belief_act = is_belief_act

    def set_is_future_tense(self, is_future_tense):
        self._is_future_tense = is_future_tense

    def has_modal(self):
        return self._has_modal
    
    def set_has_modal(self, has_modal):
        self._has_modal = has_modal

    def is_subordinate_of_said(self):
        return self._is_subordinate_of_said
    
    def is_subordinate_of_if(self):
        return self._is_subordinate_of_if

    def is_subordinate_of_then(self):
        return self._is_subordinate_of_then
    
    def has_modal_adverb(self):
        return self._has_modal_adverb
    
    def has_modal_adjective(self):
        return self._has_modal_adjective
    
    def has_negation_words(self):
        return self._has_negation_words
    
    def is_past_tense(self):
        return self._is_past_tense
    
    def is_present_tense(self):
        return self._is_present_tense
    
    def is_continuous_aspect(self):
        return self._is_continuous_aspect
    
    def is_perfect_aspect(self):
        return self._is_perfect_aspect
    
    def is_future_tense(self):
        return self._is_future_tense

    def to_dict(self):
        return {
            'is_negated': self._is_negated,
            'has_modal': self._has_modal,
            'is_subordinate_of_said': self._is_subordinate_of_said,
            'is_subordinate_of_believe': self._is_subordinate_of_believe,
            'is_subordinate_of_if': self._is_subordinate_of_if,
            'is_speech_act': self._is_speech_act,
            'is_belief_act': self._is_belief_act,
            'is_subordinate_of_expects': self._is_subordinate_of_expects,
            'is_subordinate_of_then': self._is_subordinate_of_then,
            'has_modal_adverb': self._has_modal_adverb,
            'has_modal_adjective': self._has_modal_adjective,
            'has_negation_words': self._has_negation_words,
            #'is_past_tense': self._is_past_tense,
            #'is_present_tense': self._is_present_tense,
            #'is_future_tense': self._is_future_tense,
            #'is_continuous_aspect': self._is_continuous_aspect,
            #'is_perfect_aspect': self._is_perfect_aspect,
        }

class FactualityCategorizer:
    def __init__(self):
        self._linguistic_featurizer = LinguisticFeaturizer()
        self._fns = [
            self._is_negated,
            self._has_modal,
            self._is_subordinate_of_said,
            self._is_subordinate_of_believe,
            self._is_subordinate_of_if,
            self._is_speech_act,
            self._is_belief_act,
            self._is_subordinate_of_expects,
            self._is_subordinate_of_then,
            self._has_modal_adverb,
            self._has_modal_adjective,
            self._has_negation_words,
            self._tense,
        ]

    def categorize(self, sentence, verb, required_count=0):
        fdoc = self._linguistic_featurizer.featurize_sentence(sentence.lower())
        features_array = FeaturesArray()
        counter = 0
        for token in fdoc.tokens():
            if token.text().lower() == verb.lower():
                if counter == required_count:
                    for fn in self._fns:
                        fn(token, features_array, sentence)
                counter += 1
        return features_array
    
    def _tense(self, token, features_array, sentence):
        context = sentence
        tense = None
        aspect = None
        if token is None:
            return tense, aspect
        if token.pos() in ['VERB', 'ROOT', 'AUX']:
            tense = 'Pres'
            if token.tense() is not None:
                tense = token.tense()
            aspect = token.aspect()
            aux_there = False
            if 'aux' in token.children():
                for child in token.children()['aux']:
                    if child.tense() is not None:
                        tense = child.tense()
                        if child.text() in past_perf_aux + pres_perf_aux:
                            aux_there = True
                            aspect = 'Perf'
            if aux_there is False and aspect == 'Perf':
                aspect = None
        
            paragraph = context[0]
            if any(future_modal in paragraph[max(0, token.idx() - 20): token.idx()].lower() for future_modal in future_modals):
                tense = 'Future'

        if tense == 'Past':
            features_array.set_is_past_tense(True)
        if tense == 'Pres':
            features_array.set_is_present_tense(True)
        if aspect == 'Prog':
            features_array.set_is_continuous_aspect(True)
        if aspect == 'Perf':
            features_array.set_is_perfect_aspect(True)
        if tense == 'Future':
            features_array.set_is_future_tense(True)

    def _is_negated(self, token, features_array, sentence):
        children = [token]
        while len(children) > 0:
            child = children.pop()
            if child.dep() == 'neg':
                features_array.set_is_negated(True)
            if child.text() in ['none']:
                features_array.set_is_negated(True)
            if child not in ['advcl', 'ccomp', 'xcomp']:
                self._append_children(child, children)

    def _has_modal(self, token, features_array, sentence):
        till_root = [token]
        parent = token.parent()
        while parent is not None:
            till_root.append(parent)
            parent = parent.parent()
        for token in till_root:
            children = token.children()
            for dep, dep_children in children.items():
                for child in dep_children:
                    if child.text() in future_modals:
                        features_array.set_has_modal(True)

    def _is_subordinate_of_said(self, token, features_array, sentence):
        parent = token.parent()
        while parent is not None:
            if parent.text().lower() in confident_said_verbs:
                features_array.set_is_subordinate_of_said(True)
            parent = parent.parent()

    def _is_subordinate_of_believe(self, token, features_array, sentence):
        parent = token.parent()
        while parent is not None:
            if parent.text().lower() in believe_verbs:
                features_array.set_is_subordinate_of_believe(True)
            parent = parent.parent()

    def _is_subordinate_of_if(self, token, features_array, sentence):
        till_root = [token]
        parent = token.parent()
        while parent is not None:
            till_root.append(parent)
            parent = parent.parent()
        for token in till_root:
            children = token.children()
            for dep, dep_children in children.items():
                for child in dep_children:
                    if child.text() == 'if':
                        features_array.set_is_subordinate_of_if(True)

    def _is_subordinate_of_then(self, token, features_array, sentence):
        children = [token]
        while len(children) > 0:
            child = children.pop()
            if child.text() == 'then':
                features_array.set_is_subordinate_of_then(True)
            self._append_children(child, children)

    def _is_subordinate_of_expects(self, token, features_array, sentence):
        parent = token
        while parent is not None and parent.dep() in ['xcomp']:
            if parent.parent().text().lower() in expect_verbs:
                features_array.set_is_subordinate_of_expects(True)
            parent = parent.parent()

    def _is_belief_act(self, token, features_array, sentence):
        if token.text() in believe_verbs:
            features_array.set_is_belief_act(True)

    def _is_speech_act(self, token, features_array, sentence):
        if token.text() in confident_said_verbs:
            features_array.set_is_speech_act(True)

    def _append_children(self, token, children, side='both'):
        for dep, dep_children in token.children().items():
            if dep not in complement_deps:
                for child in dep_children:
                    if side in ['left', 'both'] and child.idx() < token.idx():
                        children.append(child)

    def _has_modal_adverb(self, token, features_array, sentence):
        parent = token
        while parent is not None and parent.parent() is not None:# and parent.dep() in ['xcomp']:
            if parent.parent().text().lower() in modal_adverbs:
                features_array.set_has_modal_adverb(True)
            parent = parent.parent()

    def _has_modal_adjective(self, token, features_array, sentence):
        parent = token
        while parent is not None and parent.parent() is not None:# and parent.dep() in ['xcomp']:
            if parent.parent().text().lower() in modal_adjectives:
                features_array.set_has_modal_adjective(True)
            parent = parent.parent()

    def _has_negation_words(self, token, features_array, sentence):
        children = [token]
        first = True
        while len(children) > 0:
            child = children.pop()
            if first is False and child.text() in negation_words:
                features_array.set_has_negation_words(True)
            if first is True and child not in ['advcl', 'ccomp', 'xcomp']:
                self._append_children(child, children, 'left')
                first = False
        if token.dep() in ['pcomp'] and token.parent() is not None and token.parent().text() in negation_words:
            features_array.set_has_negation_words(True)


if __name__ == '__main__':
    fc = FactualityCategorizer()
    t1 = fc.categorize('I am going to the store', 'going')
    t2 = fc.categorize('I am not going to the store', 'going')
    t3 = fc.categorize('I may not go to the store', 'go')
    t4 = fc.categorize('Tina said, "I may not go to the store"', 'go')
    t5 = fc.categorize('Tina believes, "I may not go to the store"', 'go')
    t6 = fc.categorize('If I go to the store then I will buy candy', 'go')
    t7 = fc.categorize('Tina said, "I may not go to the store"', 'said')
    t8 = fc.categorize('Tina believes, "I may not go to the store"', 'believes')
    t9 = fc.categorize('He expects to buy the house', 'buy')
    t10 = fc.categorize('Lawyers for the relatives in Miami filed a brief in federal court on Monday arguing the Immigration and Naturalization Service can not return Elian to Cuba without holding a political asylum hearing . ', 'holding')

    print(t1.to_dict())
    print(t2.to_dict())
    print(t3.to_dict())
    print(t4.to_dict())
    print(t5.to_dict())
    print(t6.to_dict())
    print(t7.to_dict())
    print(t8.to_dict())
    print(t9.to_dict())
    print(t10.to_dict())


