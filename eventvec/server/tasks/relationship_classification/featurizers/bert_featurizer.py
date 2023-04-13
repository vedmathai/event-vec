
from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer

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


class BERTLinguisticFeaturizer:
    def __init__(self):
        self._linguistic_featurizer = LinguisticFeaturizer()

    def featurize(self, datum):
        featurized = self._linguistic_featurizer.featurize_document(
            datum.from_original_sentence()
        )
        decoded_sentence = datum.from_decoded_sentence()[0].split()
        if 'entity1' in decoded_sentence:
            entity_idx = decoded_sentence.index('entity1') + 1
            word1 = decoded_sentence[entity_idx]
            for sentence in featurized.sentences():
                for token in sentence.tokens():
                    if token.text().lower() == word1.lower():
                        closest_parent = self.closest_tense_aspect(token)
                        datum.set_from_tense(token.tense())
                        datum.set_parent_from_tense(closest_parent.tense())
                        is_future = self.check_future_tense(decoded_sentence, entity_idx)
                        if is_future is True:
                            datum.set_from_tense('FUTURE')
                        entity_idx = closest_parent.i_in_sentence()
                        is_future = self.check_future_tense(decoded_sentence, entity_idx)
                        if is_future is True:
                            datum.set_parent_from_tense('FUTURE')
                        datum.set_from_aspect(token.aspect())
                        datum.set_from_pos(token.pos())
                        datum.set_from_tag(token.tag())
                        datum.set_parent_from_aspect(closest_parent.aspect())
                        datum.set_parent_from_pos(closest_parent.pos())
                        datum.set_parent_from_tag(closest_parent.tag())
                        marked_up_parent_sentence = self.markup_parent_verb(sentence, entity_idx, 'entity1')
                        datum.set_marked_up_parent_from_sentence(marked_up_parent_sentence)
        featurized = self._linguistic_featurizer.featurize_document(
            datum.to_original_sentence()
        )
        decoded_sentence = datum.to_decoded_sentence()[0].split()
        if 'entity2' in decoded_sentence:
            entity_idx = decoded_sentence.index('entity2') + 1
            word1 = decoded_sentence[entity_idx]
            for sentence in featurized.sentences():
                for token in sentence.tokens():
                    if token.text().lower() == word1.lower():
                        closest_parent = self.closest_tense_aspect(token)
                        datum.set_to_tense(token.tense())
                        datum.set_parent_to_tense(closest_parent.tense())
                        is_future = self.check_future_tense(decoded_sentence, entity_idx)
                        if is_future is True:
                            datum.set_to_tense('FUTURE')
                        entity_idx = closest_parent.i_in_sentence()
                        is_future = self.check_future_tense(decoded_sentence, entity_idx)
                        if is_future is True:
                            datum.set_parent_to_tense('FUTURE')
                        datum.set_to_aspect(token.aspect())
                        datum.set_to_pos(token.pos())
                        datum.set_to_tag(token.tag())
                        datum.set_parent_to_aspect(closest_parent.aspect())
                        datum.set_parent_to_pos(closest_parent.pos())
                        datum.set_parent_to_tag(closest_parent.tag())
                        marked_up_parent_sentence = self.markup_parent_verb(sentence, entity_idx, 'entity2')
                        datum.set_marked_up_parent_to_sentence(marked_up_parent_sentence)

    def closest_tense_aspect(self, token):
        found, parent = token.closest_parents(['VERB', 'AUX'])
        if found is True:
            return parent
        if found is False:
            return token

    def markup_closest_verb(self, token, featurized_sentence, is_first):
        marker = 'entity1'
        if is_first is True:
            marker = 'entity2'
        new_sentence = []
        found, parent = token.closest_parents(['VERB', 'AUX'])
        for token in featurized_sentence.tokens():
            if found and parent.i() == token.i():
                new_sentence.extend([marker, token.text(), marker])
                break
            else:
                new_sentence.append(token.text())
        return new_sentence

    def check_future_tense(self, sentence, entity_idx):
        window_start = entity_idx - 6
        window_end = entity_idx
        window = sentence[window_start: window_end]
        window = ' '.join(window)
        if any(i in window for i in future_modals):
            return True
        return False

    def markup_parent_verb(self, sentence, entity_idx, replace):
        sentence = [i.text() for i in sentence.tokens()]
        sentence = sentence[:entity_idx] + [replace] + [sentence[entity_idx]] + [replace] + sentence[entity_idx+1:]
        return sentence
