from collections import defaultdict
import numpy as np

from eventvec.server.config import Config
from eventvec.server.datamodels.qa_datamodels.qa_datum import QADatum
from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.tasks.question_answering.datahandlers.datahanders_registry import DatahandlersRegistry

tense_mapping = {
    "Pres": 0,
    "Past": 1,
    'Future': 2,
    None: 3,
}

tense_num = len(tense_mapping.keys())

pos_mapping = {
    "VERB": 0,
    "AUX": 1,
    "NOUN": 2,
    "OTHERS": 3,
    None: 4,
}

pos_num = len(pos_mapping.keys())

aspect_mapping = {
    "Perf": 0,
    "Prog": 1,
    None: 2,
}

aspect_num = len(aspect_mapping.keys())

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

past_perf_aux = [
    'had',
]

pres_perf_aux = [
    'has',
    'have',
]

verb_map = {
    'VBD': 'VBD',
    'VFuture': 'VFuture',
    'VB': 'VB',
}

use_parent = True

class TenseAspectAnalyser:
    def __init__(self):
        self._datahandlers_registry = DatahandlersRegistry()

    def load(self):
        datahandler_class = self._datahandlers_registry.get_datahandler('torque')
        self._datahandler = datahandler_class()
        self._config = Config.instance()
        self._linguistic_featurizer = LinguisticFeaturizer()

        self._total_count = 0
        self._answer_count = 0
        self._eval_data = self._datahandler.qa_eval_data().data()
        self._train_data = self._datahandler.qa_train_data().data()
        self._featurized_context_cache = {}
        self._counter = defaultdict(int)
        self._example_counter = defaultdict(int)
        self._answer_counter = defaultdict(int)
        self._contexts = set()
        self._questions = set()
        self._examples = []

    def analyze(self):
        self._analyze_train_data()

    def _analyze_train_data(self):
        train_data = self._eval_data
        data_size = len(train_data)
        for datum_i, datum in enumerate(train_data):
            self._analyze_datum(datum_i, datum)
        for k, v in sorted(self._counter.items(), key=lambda x: str(x[0])):
            print(k, v)
        print('-' * 24)
        for k, v in sorted(self._answer_counter.items(), key=lambda x: str(x[0])):
            print(k, v)
        tuples = []
        for k, v in self._counter.items():
            tuples.append((k[0], k[1], v))
        mutual_information = self.calculate_mutual_information(tuples)
        info, totals = self.calculate_feature_information(tuples)
        for ii, (k, v) in enumerate(sorted(info.items(), key=lambda x: str(x[0]))):
            print('{} & {} & {} & {} & {} \\\\'.format(ii + 1, k[1][0], k[2], int(v*1000) / 1000, totals[k]))
        contexts = {}
        for ii, i in enumerate(self._examples):
            contexts[i[6]] = len(contexts)
        for ii, i in enumerate(contexts.keys()):
            print('{} & {} \\\\'.format(ii + 1, i))
        for ii, i in enumerate(self._examples):
            context = contexts[i[6]]
            print('{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(ii + 1, i[0], i[1], i[2], i[3], i[4], i[5], context, i[7]))




    def _eval_epoch(self):
        eval_data = self._eval_data
        for datum_i, datum in enumerate(eval_data):
            self._analyze_datum(datum_i, datum)

    def _analyze_datum(self, datum_i, qa_datum: QADatum):
        if qa_datum.use_in_eval() is True:
            return
        self._datum2counts(qa_datum)

    def _datum2counts(self, qa_datum):
        token_indices = []
        answer_tenses = []
        answer_aspects = []
        answer_pos = []
        required_answer = []
        question_tense = 'unknown'
        question_aspect = 'unknown'
        question_pos = 'unknown'
        context = qa_datum.context()
        answer_tenses = []
        answer_tuples = []
        bracket_context = []
        tense2count = defaultdict(int)
        if context[0] not in self._featurized_context_cache:
            self._featurized_context_cache[context[0]] = self._linguistic_featurizer.featurize_document(context[0])
        featurized_context = self._featurized_context_cache[context[0]]
        context_i2token = {}
        for sentence in featurized_context.sentences():
            for token in sentence.tokens():
                if token.entity_type() in ['TIME', 'DATE', 'EVENT']:
                    self._contexts.add(qa_datum.context()[0])
                bracket_context += [token.text()]
                if str(token.tag())[0] == 'V':
                    bracket_context += ['(',token.tense(), token.aspect(), token.tag(), ')']
            for token in sentence.tokens():
                context_i2token[token.idx()] = token
                if token.text() in qa_datum.question_events():
                    question_token = token
                    question_tense = token.tense()
                    question_aspect = token.aspect() if (token is not None) else None
                    #question_aspect = token.aspect() if (token is not None and token.text() in past_perf_aux + pres_perf_aux) else None
                    if use_parent and token is not None and 'aux' in token.children():
                        for child in token.children()['aux']:
                            if child.tense() is not None:
                                question_tense = child.tense()
                                if child.text() in past_perf_aux + pres_perf_aux:
                                    aspect = 'Perf'
                    paragraph = context[0]
                    pos = token.tag()
                    if any(future_modal in paragraph[max(0, token.idx() - 20): token.idx()].lower() for future_modal in future_modals):
                        question_tense = 'VFuture'
                    temp = token
                    while use_parent and question_tense is None and token is not None and (temp.tense() is None) and temp.parent() is not None:
                        temp = temp.parent()
                        question_tense = temp.tense()
                        question_aspect = token.aspect() if (token is not None and token.text() in past_perf_aux + pres_perf_aux ) else None
                        question_pos = token.tag() if token is not None else None
                        if any(future_modal in paragraph[max(0, token.idx() - 20): token.idx()].lower() for future_modal in future_modals):
                            question_tense = 'VFuture'
                            question_pos = 'VFuture'
                    if str(pos)[0] == 'V':
                        question_pos = verb_map.get(pos, verb_map['VB'])
                    question_pos = pos
        for answer in qa_datum.answers():
            required_answer.append(answer.text())
            for paragraph_i, paragraph in enumerate(context):
                if paragraph_i == answer.paragraph_idx():
                    if answer.start_location() is not None and answer.end_location() is not None:
                        token = context_i2token.get(answer.start_location())
                        tense = token.tense() if token is not None else None
                        temp = token

                        pos = token.tag() if token is not None else None
                        aspect = token.aspect() if (token is not None) else None
                        # aspect = token.aspect() if (token is not None and token.text() in past_perf_aux + pres_perf_aux) else None

                        if any(future_modal in paragraph[max(0, answer.start_location() - 20): answer.start_location()].lower() for future_modal in future_modals):
                            tense = 'VFuture'
                            pos = 'VFuture'

                        if use_parent and token is not None and 'aux' in token.children():
                            for child in token.children()['aux']:
                                if child.tense() is not None:
                                    tense = child.tense()
                                    if child.text() in past_perf_aux + pres_perf_aux:
                                        aspect = 'Perf'

                        
                        while use_parent and (tense is None and token is not None and (temp.tense() is None) and temp.parent() is not None):
                            temp = temp.parent()
                            tense = temp.tense()
                            aspect = token.aspect() if (token is not None and token.text() in past_perf_aux + pres_perf_aux) is not None else None
                            pos = token.tag() if token is not None else None
                            if any(future_modal in paragraph[max(0, token.idx() - 20): token.idx()].lower() for future_modal in future_modals):
                                tense = 'VFuture'
                                pos = 'VFuture'

                        answer_tenses += [tense]
                        answer_aspects += [aspect]
                        if str(pos)[0] == 'V':
                            answer_pos = verb_map.get(pos, verb_map['VB'])
                        answer_tuples += [tuple([tense, aspect, 'in'])]
                        if tense == 'VFuture':
                            self._questions.add(qa_datum.question())
                        key = (question_tense, question_aspect, tense, aspect, 'in')
                        if self._example_counter[key] <= 0 and question_tense in ['Pres', 'Past', 'VFuture'] and tense in ['Pres', 'Past', 'VFuture'] :
                            self._examples.append([question_tense, question_aspect, tense, aspect, answer.text(), qa_datum.question(), qa_datum.context()[0], 'in'])
                            self._example_counter[key] += 1
                        #if True and question_tense == 'Past' and question_aspect is None and tense == 'Past' and aspect is None and 'happened before' in qa_datum.question():

                        #    print(qa_datum.question(), question_token.text(), token.text(), ' '.join([str(i) for i in bracket_context]))
                        #    print()

        for answer in qa_datum.context_events():
            if answer.text() not in required_answer:
                for paragraph_i, paragraph in enumerate(context):
                    if paragraph_i == answer.paragraph_idx():
                        if answer.start_location() is not None and answer.end_location() is not None:
                            token = context_i2token.get(answer.start_location())
                            tense = token.tense() if token is not None else None
                            temp = token

                            pos = token.tag() if token is not None else None
                            aspect = token.aspect() if (token is not None and token.text() in past_perf_aux + pres_perf_aux) else None
                            aspect = token.aspect() if (token is not None) else None


                            if tense == 'Pres' and any(future_modal in paragraph[max(0, answer.start_location() - 20): answer.start_location()].lower() for future_modal in future_modals):
                                tense = 'VFuture'
                                pos = 'VFuture'

                            if use_parent and token is not None and 'aux' in token.children():
                                for child in token.children()['aux']:
                                    if child.tense() is not None:
                                        tense = child.tense()
                                        if child.text() in past_perf_aux + pres_perf_aux:
                                            aspect = 'Perf'

                            while use_parent and token is not None and (temp.tense() is None) and temp.parent() is not None:
                                temp = temp.parent()
                                tense = temp.tense()
                                aspect = token.aspect() if (token is not None and token.text() in past_perf_aux + pres_perf_aux) else None
                                pos = token.tag() if token is not None else None
                                if any(future_modal in paragraph[max(0, token.idx() - 20): token.idx()].lower() for future_modal in future_modals):
                                    tense = 'VFuture'
                                    pos = 'VFuture'

                            answer_tenses += [tense]
                            answer_aspects += [aspect]
                            if str(pos)[0] == 'V':
                                answer_pos = verb_map.get(pos, verb_map['VB'])
                            answer_tuples += [tuple([tense, aspect, 'out'])]
                            key = (question_tense, question_aspect, tense, aspect, 'out')
                            if self._example_counter[key] == 0 and question_tense in ['Pres', 'Past', 'VFuture'] and tense in ['Pres', 'Past', 'VFuture']:
                                self._examples.append([question_tense, question_aspect, tense, aspect, answer.text(), qa_datum.question(), qa_datum.context()[0], 'out'])
                                self._example_counter[key] += 1
                            #if False and question_tense == 'Past' and question_aspect is None and tense == 'Pres' and aspect is None and 'happened after' in qa_datum.question():
                            #    print(qa_datum.question(), question_token.text(), token.text(), '|||', ' '.join([str(i) for i in bracket_context]))
                            #    print()


        for answer_tuple in answer_tuples:
            question_tuple = tuple([question_tense])
            if True or question_tense == 'Past' and answer_tuple[0] == 'Pres':
                if 'happened after' in qa_datum.question():
                    self._counter[((question_tuple, (question_tuple[0],), 'after')), (answer_tuple[-1])] += 1
                if 'happened before' in qa_datum.question():
                    self._counter[((question_tuple, (question_tuple[0],), 'before')), (answer_tuple[-1])] += 1
                if any(i in qa_datum.question() for i in ['happened during', 'happened while']):
                    self._counter[((question_tuple, (question_tuple[0],), 'during')), (answer_tuple[-1])] += 1

        

        if 'after' in qa_datum.question():
            self._answer_counter[(len(qa_datum.answers()), 'after')] += 1
        if 'before' in qa_datum.question():
            self._answer_counter[(len(qa_datum.answers()), 'before')] += 1
        if 'during' in qa_datum.question():
            self._answer_counter[(len(qa_datum.answers()), 'during')] += 1

    def calculate_feature_information(self, tuples):
        X = defaultdict(lambda: defaultdict(int))
        info = defaultdict(int)
        totals = defaultdict(int)

        for t in tuples:
            X[t[0]][t[1]] += t[2]
        for k in X:
            total = sum(X[k].values())
            totals[k] = total
            for v in X[k]:
                X[k][v] /= total

        for k in X:
            for v in X[k]:
                info[k] -= (X[k][v] * np.log2(X[k][v]))
        return info, totals

    def calculate_mutual_information(self, tuples):
        X = defaultdict(int)
        Y = defaultdict(int)
        XY = defaultdict(int)
        for t in tuples:
            X[t[0]] += t[2]
            Y[t[1]] += t[2]
            XY[(t[0], t[1])] += t[2]
        X_sum = sum(X.values())
        Y_sum = sum(Y.values())
        XY_sum = sum(XY.values())
        total = 0
        for x in X.keys():
            px = float(X[x]) / X_sum
            for y in Y.keys():
                py = float(Y[y]) / Y_sum
                pxy = float(XY[(x, y)]) / XY_sum
                if pxy != 0:
                    total += pxy * np.log((pxy)/px / py)
        return total

if __name__ == '__main__':
    qa_train = TenseAspectAnalyser()
    qa_train.load()
    qa_train.analyze()


### Look to see whether the correct attribution of tense and aspect is happening from the dependency parse, maybe the correct set of verbs will have to be found.
### Do error analysis of why the tense and aspect and relationships are not matching up
