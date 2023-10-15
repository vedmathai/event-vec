import re

from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.common.lists.said_verbs import said_verbs
from eventvec.server.utils.general import token2parent, token2tense

tag2tense = {
    "VB": "Pres",
    "VBD": "Past",
    "VBG": "Pres",
    "VBN": "Past",
    "VBP": "Pres",
    "VBZ": "Pres",
}

future_modals = [
    'will',
    'going to',
    'would',
    'could',
    'might',
    'may',
    'can',
    'going to',
]


past_perf_aux = ['had']
pres_perf_aux = ['has', 'have']

future_modals = [
    'will',
    'going to',
    'would',
    'could',
    'might',
    'may',
    'can',
    'going to',
]


class BERTLinguisticFeaturizer:
    def __init__(self):
        self._linguistic_featurizer = LinguisticFeaturizer()

    def featurize(self, datum):
        featurized = self._linguistic_featurizer.featurize_document(
            datum.from_original_sentence()
        )
        parent_1 = None
        parent_2 = None
        predecoded_sentence = datum.from_decoded_sentence()[0]
        if 'entity1' in predecoded_sentence:
            decoded_sentence = re.sub('entity1', '', predecoded_sentence)
            decoded_sentence = decoded_sentence.split()
            entity_idx = predecoded_sentence.split().index('entity1')
            word1 = decoded_sentence[entity_idx]
            for sentence in featurized.sentences():
                for token in sentence.tokens():
                    if token.text().lower() == word1.lower():
                        tense, aspect = self.token2tense(datum.from_original_sentence(), token)
                        parent = self.token2parent(token)
                        parent_1 = parent
                        parent_tense, parent_aspect = self.token2tense(datum.from_original_sentence(), parent)
                        datum.set_from_tense(tense)
                        datum.set_from_aspect(aspect)
                        datum.set_parent_from_tense(parent_tense)
                        datum.set_parent_from_aspect(parent_aspect)

        featurized = self._linguistic_featurizer.featurize_document(
            datum.to_original_sentence()
        )
        predecoded_sentence = datum.to_decoded_sentence()[0]
        if 'entity2' in predecoded_sentence:
            decoded_sentence = re.sub('entity2', '', predecoded_sentence)
            decoded_sentence = decoded_sentence.split()
            entity_idx = predecoded_sentence.split().index('entity2')
            word2 = decoded_sentence[entity_idx]
            for sentence in featurized.sentences():
                for token in sentence.tokens():
                    if token.text().lower() == word2.lower():
                        tense, aspect = self.token2tense(datum.to_original_sentence(), token)
                        parent = self.token2parent(token)
                        parent_2 = parent
                        parent_tense, parent_aspect = self.token2tense(datum.to_original_sentence(), parent)
                        datum.set_to_tense(tense)
                        datum.set_to_aspect(aspect)
                        datum.set_parent_to_tense(parent_tense)
                        datum.set_parent_to_aspect(parent_aspect)

        if datum.from_original_sentence() == datum.to_original_sentence():
                
            if parent_1 is not None and parent_1.text() == word2.lower():
                datum.set_is_interested(True)

            if parent_2 is not None and parent_2.text() == word1.lower():
                datum.set_is_interested(True)

    def token2tense(self, sentence, token):
        sentence = ' '.join(sentence)
        return token2tense(sentence, token)
    
    def token2parent(self, token):
        return token2parent(None, token)
    
