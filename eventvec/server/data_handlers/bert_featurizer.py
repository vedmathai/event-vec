from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer


class BERTLinguisticFeaturizer:
    def __init__(self):
        self._linguistic_featurizer = LinguisticFeaturizer()

    def featurize(self, datum):
        featurized = self._linguistic_featurizer.featurize_sentence(
            datum.from_original_sentence()
        )
        decoded_sentence = datum.from_decoded_sentence()[0].split()
        if 'entity1' in decoded_sentence:
            entity_idx = decoded_sentence.index('entity1') + 1
            word1 = decoded_sentence[entity_idx]
            for token in featurized.tokens():
                if token.text().lower() == word1.lower():
                    closest_parent = self.closest_tense_aspect(token)
                    datum.set_from_tense(token.tense())
                    datum.set_from_aspect(token.aspect())
                    datum.set_from_pos(token.pos())
        featurized = self._linguistic_featurizer.featurize_sentence(
            datum.to_original_sentence()
        )
        decoded_sentence = datum.to_decoded_sentence()[0].split()
        if 'entity2' in decoded_sentence:
            entity_idx = decoded_sentence.index('entity2') + 1
            word1 = decoded_sentence[entity_idx]
            for token in featurized.tokens():
                if token.text().lower() == word1.lower():
                    closest_parent = self.closest_tense_aspect(token)
                    datum.set_to_tense(closest_parent.tense())
                    datum.set_to_aspect(closest_parent.aspect())
                    datum.set_to_pos(token.pos())

    def closest_tense_aspect(self, token):
        found, parent = token.closest_children(['VERB', 'AUX'])
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
            else:
                new_sentence.append(token.text())
        return new_sentence
