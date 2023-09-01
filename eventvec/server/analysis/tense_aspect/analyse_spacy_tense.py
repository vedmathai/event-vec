from collections import defaultdict
import numpy as np

from eventvec.server.config import Config
from eventvec.server.datamodels.qa_datamodels.qa_datum import QADatum
from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.tasks.question_answering.datahandlers.datahanders_registry import DatahandlersRegistry

past_perf_aux = ['had']
pres_perf_aux = ['has', 'have']

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
					  "comment", "comments", "commented", "uphold", "upholds", "upheld", "rule", "ruled", "ruling", "look", "looked", "looking",
                      "announced", "cited", "quoted", "telling", "continued", "replied", "derided", "declined", "estimates", "urges", "quipped",
                      "recommends", "denounced", "recalled", "recommended"
                      'rule', 'ruled', 'ruling', 'look', 'looked', 'looking',
                      'continue', 'continued', 'continuing',
                      'lies', 'lying', 'lied',
                      'replied', 'replies', 'replied',
                      'heard', 'hears', 'hearing',
                      'adds', 'added', 'adding',
                       'estimates', 'estimated', 'estimating',
                      'promised', 'promise', 'promising',
                      'hoped', 'hoping', 'hopes', 'hope',
                      'accused', 'accusing', 'accuses', 
                      'urges', 'urged', 'urging', 
                      'stipulates', 'stipulated', 'stipulating',
                      'speculated', 'speculates', 'speculating', 
                      'assured', 'assuring', 'assures',
                      'predicted', 'predicts', 'predicting',
                      'announced', 'announces', 'announcing',
                      'cited', 'citing', 'cites',
                      'portends', 'portending', 'portended',
                      'recommends', 'recommending', 'recommended',
                      'quipped', 'quipping', 'quips',
                      'criticised', 'criticising', 'critises',
                      'reassured', 'reassuring', 'reassures',
                      'quoted', 'quotes', 'quoting', 
                      'demands', 'demanded', 'demanding', 
                      'replied', 'replies', 'replying',
                      'denounced', 'denouncing', 'denounces',
                      'knowing', 'knowed', 'knows',
                      'reiterated', 'reiterates', 'reiterating', 
                      'reading', 'read',
                      'questions', 'questioning', 'questioned',
                      'arguing', 'argued', 'argues',
                      'signalled', 'signals', 'signalling',
                      'accuse', 'accusing', 'accused', 
                      'hinted', 'hints', 'hinting',
                      'questioned', 'questioning', 'questions',
                      'asked', 'asking', 'asks',
                      'tells', 'told', 'telling',
                      'vowed', 'vows', 
                      'urged', 'urging', 'urges',
])
                      
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
        for key, value in self._counter.items():
            print(key, value)

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
                tense, aspect = self.token2tense(qa_datum, token)
                parent_token = self.token2parent(qa_datum, token)
                parent_tense, parent_aspect = self.token2tense(qa_datum, parent_token)

                if parent_token is not None and parent_token.text() in said_verbs:
                    if token.text() in qa_datum.question_events() and parent_token.text() in [i.text() for i in qa_datum.answers()]:
                        altered_required.append(parent_token.text())
                        print(qa_datum.question())
                        print(context)
                        print('token', token.text(), tense, aspect, 'parent', parent_token.text(), parent_tense, parent_aspect)
                        print(qa_datum.question_events(), [i.text() for i in qa_datum.answers()])
                        print(' ')
                    if token.text() in qa_datum.question_events():
                        possible_answers.append(parent_token.text())

                    if parent_token.text() in qa_datum.question_events() and token.text() in [i.text() for i in qa_datum.answers()]:
                        altered_required.append(token.text())
                    if parent_token.text() in qa_datum.question_events():
                        possible_answers.append(token.text())

                #if parent_tense is not None and (parent_token.text() in qa_datum.question_events() or token.text() in qa_datum.question_events()):
                #    print('\n'  *3)
                #    print(qa_datum.context(), qa_datum.question(), 'parent:', parent_token.text(), 'token:', token.text(), qa_datum.question_events())
                #    
                if tense is not None and parent_tense is not None:
                    self._counter[(parent_tense, tense)] += 1
                if tense is not None:
                    annotated_context.append('{} ({} {})'.format(token.text(), tense, aspect))
                else:
                    annotated_context.append(token.text())
            #print(' '.join(annotated_context))
            #print('\n' * 4)
        self._counts_counter[parent_counter] += 1

        self._example_counter += 1
            

    def token2tense(self, qa_datum, token):
        context = qa_datum.context()
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
        return tense, aspect

    def token2parent(self, qa_datum, token):
        deps = ['ccomp', 'xcomp', "parataxis", '-relcl', 'conj']
        parent = None
        use = False
        if token.dep() in deps:
            parent = token.parent()
            while not (parent is None or parent.text() in said_verbs or parent.dep() == 'ROOT'):
                if (token.dep() in deps and token.pos() in ['VERB']) or parent.dep() in ['ccomp', 'xcomp']:
                    use = True
                parent = parent.parent()
        if use is False or (parent is not None and parent.text() not in said_verbs):
            parent = None
        return parent


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