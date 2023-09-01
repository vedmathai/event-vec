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
    'to be',
]

said_verbs = set(["observe", "observes", "observed", "describe", "describes", "described", "discuss", "discusses", "discussed",
					  "report", "reports", "reported", "outline", "outlines", "outlined", "remark", "remarks", "remarked", 	
					  "state", "states", "stated", "go on to say that", "goes on to say that", "went on to say that", 	
					  "quote that", "quotes that", "quoted that", "say", "says", "said", "mention", "mentions", "mentioned",
					  "articulate", "articulates", "articulated", "write", "writes", "wrote", "relate", "relates", "related",
					  "convey", "conveys", "conveyed", "recognise", "recognises", "recognised", "clarify", "clarifies", "clarified",
					  "acknowledge", "acknowledges", "acknowledged", "concede", "concedes", "conceded", "accept", "accepts", "accepted",
					  "refute", "refutes", "refuted", "uncover", "uncovers", "uncovered", "admit", "admits", "admitted",
					  "demonstrate", "demonstrates", "demonstrated", "highlight", "highlights", "highlighted", "illuminate", "illuminates", "illuminated", 							  
                      "support", "supports", "supported", "conclude", "concludes", "concluded", "elucidate", "elucidates", "elucidated",
					  "reveal", "reveals", "revealed", "verify", "verifies", "verified", "argue", "argues", "argued", "reason", "reasons", "reasoned",
					  "maintain", "maintains", "maintained", "contend", "contends", "contended", 
					    "feel", "feels", "felt", "consider", "considers", "considered", 						  
                      "assert", "asserts", "asserted", "dispute", "disputes", "disputed", "advocate", "advocates", "advocated",
					  "opine", "opines", "opined", "think", "thinks", "thought", "imply", "implies", "implied", "posit", "posits", "posited",
					  "show", "shows", "showed", "illustrate", "illustrates", "illustrated", "point out", "points out", "pointed out",
					  "prove", "proves", "proved", "find", "finds", "found", "explain", "explains", "explained", "agree", "agrees", "agreed",
					  "confirm", "confirms", "confirmed", "identify", "identifies", "identified", "evidence", "evidences", "evidenced",
					  "attest", "attests", "attested", "believe", "believes", "believed", "claim", "claims", "claimed", "justify", "justifies", "justified", 							  
                      "insist", "insists", "insisted", "assume", "assumes", "assumed", "allege", "alleges", "alleged", "deny", "denies", "denied",
					   "disregard", "disregards", "disregarded", 
					   "surmise", "surmises", "surmised", "note", "notes", "noted",
					  "suggest", "suggests", "suggested", "challenge", "challenges", "challenged", "critique", "critiques", "critiqued",
					  "emphasise", "emphasises", "emphasised", "declare", "declares", "declared", "indicate", "indicates", "indicated",
					  "comment", "comments", "commented", "uphold", "upholds", "upheld"])

future_said_verbs = set([
    'anticipate', 'anticipates', 'anticipated', "hypothesise", "hypothesises", "hypothesised", "propose", "proposes", "proposed", "theorise", "theorises", "theorised", "posit", "posits", "posited",
    "speculate", "speculates", "speculated", "suppose", "supposes", "supposed", "conjecture", "conjectures", "conjectured", "envisioned", "envision", "envisions", "forecasts", 'foresee', 'forecast', 'forecasted',
    'foresaw', 'estimate', 'estimated', 'estimates'
])

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
        train_data = self._train_data
        data_size = len(train_data)
        for datum_i, datum in enumerate(train_data):
            self._analyze_datum(datum_i, datum)
        for k, v in sorted(self._counter.items(), key=lambda x: str(x[0])):
            pass
            # print(k, v)
        print('-' * 24)
        for k, v in sorted(self._answer_counter.items(), key=lambda x: str(x[0])):
            pass
            #print(k, v)
        tuples = []
        for k, v in self._counter.items():
            tuples.append((k[0], k[1], v))
        mutual_information = self.calculate_mutual_information(tuples)
        info, totals = self.calculate_feature_information(tuples)
        for ii, (k, v) in enumerate(sorted(info.items(), key=lambda x: totals[x[0]])):
            pass
            #print('info {} & {} & {} & {} & {} & {} & {} \\\\'.format(ii + 1, k[0][0], k[0][1], k[1][0], k[1][1], int(v*1000) / 1000, totals[k]))
        contexts = {}
        for ii, i in enumerate(self._examples):
            contexts[i[6]] = len(contexts)
        for ii, i in enumerate(contexts.keys()):
            pass
            #print('{} & {} \\\\'.format(ii + 1, i))
        for ii, i in enumerate(sorted(self._examples, key=lambda x: x[7])):
            context = contexts[i[6]]
            #print('{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(ii + 1, i[0], i[1], i[2], i[3], i[4], i[5], context, i[7]))




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
        self._tokeni2tense = defaultdict()

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
                    tense, aspect = self.question_tense_aspect(token, context)
                    question_tense = tense
                    question_aspect = aspect

        for answer in qa_datum.answers():
            required_answer.append(answer.text())
            for paragraph_i, paragraph in enumerate(context):
                if paragraph_i == answer.paragraph_idx():
                    if answer.start_location() is not None and answer.end_location() is not None:
                        token = context_i2token.get(answer.start_location())
                        tense, aspect = self.question_tense_aspect(token, context)
                        answer_tenses += [tense]
                        answer_aspects += [aspect]
                        answer_tuples += [tuple([tense, aspect, 'in'])]
                        key = (question_tense, question_aspect, tense, aspect, 'in')

        for answer in qa_datum.context_events():
            if answer.text() not in required_answer:
                for paragraph_i, paragraph in enumerate(context):
                    if paragraph_i == answer.paragraph_idx():
                        if answer.start_location() is not None and answer.end_location() is not None:
                            token = context_i2token.get(answer.start_location())
                            tense, aspect = self.question_tense_aspect(token, context)
                            answer_tenses += [tense]
                            answer_aspects += [aspect]
                            answer_tuples += [tuple([tense, aspect, 'out'])]
                            key = (question_tense, question_aspect, tense, aspect, 'out')


        for answer_tuple in answer_tuples:
            question_tuple = tuple([question_tense, question_aspect])
            if True or question_tense == 'Past' and answer_tuple[0] == 'Pres':
                if 'happened after' in qa_datum.question():
                    self._counter[((question_tuple, (answer_tuple[0], answer_tuple[1]), 'after')), (answer_tuple[-1])] += 1
                if 'happened before' in qa_datum.question():
                    self._counter[((question_tuple, (answer_tuple[0], answer_tuple[1]), 'before')), (answer_tuple[-1])] += 1
                if any(i in qa_datum.question() for i in ['happened during', 'happened while']):
                    self._counter[((question_tuple, (answer_tuple[0], answer_tuple[1]), 'during')), (answer_tuple[-1])] += 1

        
        if 'after' in qa_datum.question():
            self._answer_counter[(len(qa_datum.answers()), 'after')] += 1
        if 'before' in qa_datum.question():
            self._answer_counter[(len(qa_datum.answers()), 'before')] += 1
        if 'during' in qa_datum.question():
            self._answer_counter[(len(qa_datum.answers()), 'during')] += 1


    def question_tense_aspect(self, token, context):
        tense = None
        aspect = None
        if token is not None:
            #tense = 'Pres'
            tense = None

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
                    if child.text() == 'to':
                        parent = token.parent()
                        if parent is not None and parent.idx() in self._tokeni2tense:
                            tense, aspect = self._tokeni2tense[parent.idx()]


            if aux_there is False and aspect == 'Perf':
                aspect = None
        
            paragraph = context[0]
            if any(future_modal in paragraph[max(0, token.idx() - 20): token.idx()].lower() for future_modal in future_modals):
                tense = 'Future'
            #if token.text() == 'be' and 'to' in [i.text() for i in token.all_children()]:
            #    tense = 'Future'

            parent = token.parent()
            parent_tense = None
            parent_aspect = None
            if parent is not None and parent.text() in said_verbs and token.dep() in ['parataxis', 'ccomp']:
                parent_tense = parent.tense()
                parent_aspect = parent.aspect()

            if parent is not None and parent.text() in future_said_verbs and token.dep() in ['parataxis', 'ccomp']:
                parent_tense = 'Future'
                parent_aspect = parent.aspect()
            
            if parent_tense is not None and token.parent() is not None and token.tag() == 'VBG' and token.parent().idx() in self._tokeni2tense:
                parent_tense, parent_aspect = self._tokeni2tense[token.parent().idx()]

            self._tokeni2tense[token.idx()] = (tense, aspect) 
            tense = '{}{}'.format(parent_tense, tense)
            aspect = '{}{}'.format(parent_aspect, aspect)
            
        return tense, aspect


    def calculate_feature_information(self, tuples):
        X = defaultdict(lambda: defaultdict(int))
        info = defaultdict(int)
        totals = defaultdict(int)
        weighted_average = 0

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
        numer = 0
        denom = 0
        for k in info:
            numer += info[k] * totals[k]
            denom += totals[k]
        print('total', numer/denom)
        return info, totals

    def calculate_mutual_information(self, tuples):
        X = defaultdict(int)
        Y = defaultdict(int)
        XY = defaultdict(int)
        info = defaultdict(int)
        info_marginal = defaultdict(int)
        for t in tuples:
            X[tuple(t[0][0:2])] += t[2]
            Y[tuple(t[0][2:])] += t[2]
            XY[(tuple(t[0][0:2]), tuple(t[0][2:]))] += t[2]
        X_sum = sum(X.values())
        Y_sum = sum(Y.values())
        XY_sum = sum(XY.values())
        for x in X.keys():
            px = float(X[x]) / X_sum
            for y in Y.keys():
                py = float(Y[y]) / Y_sum
                pxy = float(XY[(x, y)]) / XY_sum
                if pxy != 0:
                    info[(x, y)] = pxy * (np.log2(pxy) - np.log2(XY_sum) - np.log2(px) + np.log2(X_sum) - np.log2(py) + np.log(Y_sum))
                    info_marginal[x] += info[(x, y)]
        #for key, value in sorted(info.items(), key=lambda x: str(x[0])):
        #    print('-->', key, value)
        return info


if __name__ == '__main__':
    qa_train = TenseAspectAnalyser()
    qa_train.load()
    qa_train.analyze()


### Look to see whether the correct attribution of tense and aspect is happening from the dependency parse, maybe the correct set of verbs will have to be found.
### Do error analysis of why the tense and aspect and relationships are not matching up
