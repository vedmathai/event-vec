from collections import defaultdict
import numpy as np

from eventvec.server.config import Config
from eventvec.server.datamodels.qa_datamodels.qa_datum import QADatum
from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.tasks.question_answering.datahandlers.datahanders_registry import DatahandlersRegistry

from eventvec.server.common.lists.said_verbs import said_verbs, future_said_verbs
from eventvec.server.utils.general import token2parent, token2tense

said_verbs = said_verbs | future_said_verbs

past_perf_aux = ['had']
pres_perf_aux = ['has', 'have']


class SpacyTenseAspectAnalyser:
    def __init__(self):
        self._datahandlers_registry = DatahandlersRegistry()

    def load(self):
        datahandler_class = self._datahandlers_registry.get_datahandler('torque')
        self._datahandler = datahandler_class()
        self._config = Config.instance()
        self._linguistic_featurizer = LinguisticFeaturizer()

        self._eval_data = self._datahandler.qa_eval_data().data()
        self._train_data = self._datahandler.qa_train_data().data()
        self._featurized_context_cache = {}
        self._seen = set()
        self._counter = defaultdict(int)
        self._example_counter = 0
        self._words = set()
        self._counts_counter = defaultdict(int)

    def analyze(self):
        self._analyze_train_data()
        for key, value in sorted(self._counter.items(), key = lambda x: str(x[0])):
            print(key, value)
        print(sum(self._counter.values()))

    def _analyze_train_data(self):
        train_data = self._train_data
        data_size = len(train_data)
        for datum_i, datum in enumerate(train_data):
            self._analyze_datum(datum_i, datum)
        print(self._counts_counter)
        print(self._example_counter)

    def _eval_epoch(self):
        eval_data = self._eval_data
        for datum_i, datum in enumerate(eval_data):
            self._analyze_datum(datum_i, datum)

    def _analyze_datum(self, datum_i, qa_datum: QADatum):
        if qa_datum.use_in_eval() is True:
            return
        self._datum2counts(qa_datum)

    def _datum2counts(self, qa_datum):
        context = qa_datum.context()
        if context[0] not in self._featurized_context_cache:
            self._featurized_context_cache[context[0]] = self._linguistic_featurizer.featurize_document(context[0])
        featurized_context = self._featurized_context_cache[context[0]]
        context_i2token = {}
        parent_counter = 0
        for sentence in featurized_context.sentences():
            #if sentence in self._seen:
            #    continue
            self._seen.add(sentence)
            annotated_context = []
            altered_required = []
            possible_answers = []

            for token in sentence.tokens():
                tense = None
                aspect = None
                context_i2token[token.idx()] = token
                tense, aspect = token2tense(sentence.text(), token)
                parent_token = token2parent(sentence.text(), token)
                parent_tense, parent_aspect = token2tense(sentence.text(), parent_token)

                if parent_token is not None and parent_token.text() in said_verbs:
                    is_direct_quote = False
                    for dep, tokens in parent_token.children().items():
                        for t in tokens:
                            if t.text() == '"' or t.text() == "'":
                                is_direct_quote = True
                    if token.text() in qa_datum.question_events() and parent_token.text() in [i.text() for i in qa_datum.answers()]:
                        altered_required.append(parent_token.text())
                        if 'after' in qa_datum.question():
                            self._counter[(parent_tense, parent_aspect, tense, aspect, is_direct_quote, 'after')] += 1
                        if 'before' in qa_datum.question():
                            self._counter[(parent_tense, parent_aspect, tense, aspect, is_direct_quote, 'before')] += 1  
                        if any(i in qa_datum.question() for i in ['during', 'while']):
                            self._counter[(parent_tense, parent_aspect, tense, aspect, is_direct_quote, 'during')] += 1  
                    if token.text() in qa_datum.question_events():
                        possible_answers.append(parent_token.text())
                        self._counter['possible'] += 1

                    if parent_token.text() in qa_datum.question_events() and token.text() in [i.text() for i in qa_datum.answers()]:
                        altered_required.append(token.text())
                        if 'after' in qa_datum.question():
                            self._counter[(parent_tense, parent_aspect, tense, aspect, is_direct_quote, 'before')] += 1
                        if 'before' in qa_datum.question():
                            self._counter[(parent_tense, parent_aspect, tense, aspect, is_direct_quote, 'after')] += 1  
                        if any(i in qa_datum.question() for i in ['during', 'while']):
                            self._counter[(parent_tense, parent_aspect, tense, aspect, is_direct_quote, 'during')] += 1                           
                    if parent_token.text() in qa_datum.question_events():
                        possible_answers.append(token.text())
                        self._counter['possible'] += 1

                #if parent_tense is not None and (parent_token.text() in qa_datum.question_events() or token.text() in qa_datum.question_events()):
                #    print('\n'  *3)
                #    print(qa_datum.context(), qa_datum.question(), 'parent:', parent_token.text(), 'token:', token.text(), qa_datum.question_events())
                #    
                if tense is not None:
                    annotated_context.append('{} ({} {})'.format(token.text(), tense, aspect))
                else:
                    annotated_context.append(token.text())
            #print(' '.join(annotated_context))
            #print('\n' * 4)
        self._counts_counter[parent_counter] += 1

        self._example_counter += 1
            

if __name__ == '__main__':
    qa_train = SpacyTenseAspectAnalyser()
    qa_train.load()
    qa_train.analyze()


### Look to see whether the correct attribution of tense and aspect is happening from the dependency parse, maybe the correct set of verbs will have to be found.
### Do error analysis of why the tense and aspect and relationships are not matching up
# It is a sub-clause of said etc
# Projected to, due to etc
# to be is a future
# Gerund takes the tense of the parent
# Noun events
# Temporal adverbials
# The scheduled post
# The post is scheduled for
# event-event adverbials like following, after
# prompting the coup, causing the disturbance
# finished
# Dates and times

# See how much of the dataset is following tenses and aspects of said - verb pair
# See how much of the LLMs are following the tenses and aspects of said verb pair
# See how much of the Neuro-symbolic are following the tenses and aspects of said - verb pair
# See what the errors are because of in both of them.