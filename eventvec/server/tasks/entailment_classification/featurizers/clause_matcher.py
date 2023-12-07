from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.datamodels.featurized_document_datamodel.featurized_sentence import FeaturizedSentence  # noqa

sentence_split_deps = ['conj', 'ccomp', 'xcomp', 'appos', 'advcl']
verb_pos = ['VERB', 'AUX']


class ClauseMatcher:
    def __init__(self):
        self._linguistic_featurizer = LinguisticFeaturizer()

    def match(self, sentence_1, sentence_2):
        clauses = self.splitter(sentence_1)
        sentence_2_set = set(sentence_2.split())
        max_jaccard = 0
        event = None
        for clause in clauses:
            clause_set = set([i.text() for i in clause])
            jaccard = self.jaccard(clause_set, sentence_2_set)
            if jaccard >= max_jaccard:
                max_jaccard = jaccard
                max_clause = clause
        for token in max_clause:
            if token.pos() in verb_pos:
                event = token.text()
        if event is None:
            event = sentence_2.split()[0]
        self._featurized_sentence_2 = self._linguistic_featurizer.featurize_sentence(sentence_2)
        event_2 = self._featurized_sentence_2.tokens()[0]
        for token in self._featurized_sentence_2.tokens():
            if token.pos() in verb_pos:
                event_2 = token
        return event, event_2.text()

    def jaccard(self, set_1, set_2):
        return len(set_1.intersection(set_2)) / len(set_1.union(set_2))

    def splitter(self, sentence):
        tokens = {}
        clauses = []
        self._featurized_sentence = self._linguistic_featurizer.featurize_sentence(sentence)
        for token_1 in self._featurized_sentence.tokens():
            for token_2 in self._featurized_sentence.tokens():
                dep_path = FeaturizedSentence.dependency_path_between_tokens(token_1, token_2)
                if len(dep_path) == 2:
                    if token_1.pos() in verb_pos and token_2.pos() in verb_pos:
                        if token_2.dep() in sentence_split_deps:
                            tokens[token_1.i()] = token_1
                            tokens[token_2.i()] = token_2
        for token in tokens.values():
            clauses.append(self._find_children(token))
        if len(clauses) == 0:
            return [self._featurized_sentence.tokens()]
        return clauses

    def _find_children(self, token):
        clause = [token]
        queue = [token]
        while len(queue) > 0:
            top = queue.pop(0)
            for dep, children in top.children().items():
                if dep not in sentence_split_deps:
                    queue.extend(children)
                    clause.extend(children)
        return sorted(clause, key=lambda x: x.i_in_sentence())


if __name__ == '__main__':
    cm = ClauseMatcher()
    sentence1 = "The activities included in the Unified Agenda are, in general, those expected to have a regulatory action within the next 12 months, although agencies may include activities with an even longer time frame."
    sentence2 = "Most of the activities taken under the regulatory actions have been longer that 12 months."
    m = cm.match(sentence1,  sentence2)
    print(m)