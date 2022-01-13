
from eventvec.utils.spacy_utils.utils import get_spacy_doc


class CoreferenceResolver:
    
    def resolve(self, document):
        text = document.text()
        text = get_spacy_doc(text)
        token2reference = {}
        for chain in text._.coref_chains:
            for token in chain:
                indexes = token.token_indexes
                for index in indexes:
                    token2reference[index] = text._.coref_chains.resolve(text[index])

        new_doc = []
        for token in text:
            if token.i in token2reference and token2reference[token.i] is not None:
                new_doc += token2reference[token.i]
            else:
                new_doc += [token]

        new_doc_text = [i.orth_ for i in new_doc]
        new_doc_text = ' '.join(new_doc_text)
        document.set_resolved_text(new_doc_text)
        return document
