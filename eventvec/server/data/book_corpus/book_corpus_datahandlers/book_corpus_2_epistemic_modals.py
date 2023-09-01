import json

from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer  # noqa
from eventvec.server.data.book_corpus.book_corpus_datahandlers.book_corpus_llm_datahandler import BookCorpusDatahandler  # noqa

auxs = {'can', 'could', 'would', 'will', 'wo', 'must', 'll', 'should', 'shall', 'might', 'may', "'d"}
                
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

future_said_verbs = set([
    'anticipate', 'anticipates', 'anticipated',
    "hypothesise", "hypothesises", "hypothesised",
    "propose", "proposes", "proposed", "theorise", "theorises", "theorised", "posit", "posits", "posited",
    "speculate", "speculates", "speculated", "suppose", "supposes", "supposed", "conjecture", "conjectures", "conjectured", "envisioned", "envision", "envisions", "forecasts", 'foresee', 'forecast', 'forecasted',
    'foresaw', 'estimate', 'estimated', 'estimates'
])

said_verbs = said_verbs | future_said_verbs

adverbs = set(["probably", "possibly", "clearly", "obviously",
               "presumably", "evidently", "apparently", "supposedly",
               "conceivably", "undoubtedly", "allegedly", "reportedly",
               "arguably", "unquestionably", "seemingly", "certainly",
               "likely", "credibly", "feasibly", "plausably", "reasonably"
               "believably", "seemingly", "surely", "precisely", "plainly",
               "obviously", "definitely", "distinctly", "undeniably",
               "unmistakably", "undeniably", "transparently", "recognizably",
               "perceptibly", "patently", "overtly", "noticeably", "markedly", "manifestly",
               "lucidly", "indubitably", "incontroverbly", "incotestably", "discernibly", "conspicously",
               ])

class BookCorpus2EpistemicModals():
    def __init__(self):
        self._book_corpus_data_handler = BookCorpusDatahandler()
        self._linguistic_featurizer = LinguisticFeaturizer()

    def load(self):
        filenames = self._book_corpus_data_handler.book_corpus_file_list()
        end = 20000
        aux_set = set()
        said_counter = 0
        aux_counter = 0
        verb_counter = 0
        aux_in_said = 0
        verb_in_said = 0
        adverb_counter = 0
        adverb_in_said = 0

        for filenamei, filename in enumerate(filenames[:end]):
            print(filenamei)
            file_contents = self._book_corpus_data_handler.read_file(filename)  # noqa
            file_contents = file_contents[:1000000-1]
            fdoc = self._linguistic_featurizer.featurize_document(file_contents)
            for fsent in fdoc.sentences():
                for token in fsent.tokens():
                    if token.pos() in ['VERB', 'AUX']:
                        if token.text() in said_verbs:
                            said_counter += 1
                        verb_counter += 1
                        for childk, childv in token.children().items():
                            if childk == 'aux':
                                for v in childv:
                                    if v.text() in auxs:
                                        aux_counter += 1
                                        verb_counter -= 1
                                        parent = v.parent()
                                        while parent is not None:
                                            if parent.dep() in ['ccomp', 'xcomp'] and parent.parent().text() in said_verbs:
                                                aux_in_said += 1
                                            parent = parent.parent()
                            if childk == 'advmod':
                                for v in childv:
                                    if v.text() in adverbs:
                                        adverb_counter += 1
                                        parent = v.parent()
                                        while parent is not None:
                                            if parent.dep() in ['ccomp', 'xcomp'] and parent.parent().text() in said_verbs:
                                                adverb_in_said += 1
                                            parent = parent.parent()
                        parent = token.parent()
                        while parent is not None:
                            if parent.dep() in ['ccomp', 'xcomp'] and parent.parent().text() in said_verbs:
                                verb_in_said += 1
                            parent = parent.parent()
            print(
                  aux_counter, verb_counter,
                 'aux_counter/verb_counter', aux_counter/verb_counter,
                 'aux_in_said/aux_counter', aux_in_said/aux_counter, 
                 'aux_in_said/verb_counter', aux_in_said/verb_counter,
                 'aux_in_said/verb_in_said', aux_in_said/(verb_in_said - aux_in_said),
                 'adverb_counter/verb_counter', adverb_counter/verb_counter,
                 'adverb_in_said/adverb_counter', adverb_in_said/adverb_counter,
                 'adverb_in_said/verb_counter', adverb_in_said/verb_counter,
                 'adverb_in_said/verb_in_said', adverb_in_said/verb_in_said,
                 'said_counter/verb_counter', said_counter/verb_counter, 
                 )


if __name__ == "__main__":
    bcld = BookCorpus2EpistemicModals()
    bcld.load()