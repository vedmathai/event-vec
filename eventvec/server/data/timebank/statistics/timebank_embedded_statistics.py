import pprint
from collections import defaultdict
import numpy as np


from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.common.lists.said_verbs import said_verbs
from eventvec.server.data.timebank.timebank_reader.timebank_dense_reader import TimeBankDenseDataReader  # noqa
from eventvec.server.data.timebank.timebank_reader.te3_gold_reader import TE3GoldDatareader  # noqa
from eventvec.server.data.matres.matres_readers.matres_reader import MatresDataReader  # noqa
from eventvec.server.data.timebank.timebank_reader.te3_silver_reader import TE3SilverDatareader  # noqa
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_event import TimebankEvent  # noqa



rel2rel = {
    'IS_INCLUDED': 'is_included',
    'SIMULTANEOUS': 'simultaneous',
    'BEFORE': 'before',
    'IDENTITY': 'identity',
    'DURING': 'during',
    'ENDED_BY': 'ended_by',
    'BEGUN_BY': 'begun_by',
    'IAFTER': 'i_after',
    'AFTER': 'after',
    'IBEFORE': 'i_before',
    'ENDS': 'ends',
    'INCLUDES': 'includes',
    'DURING_INV': 'during_inv',
    'BEGINS': 'begins',
    'EQUAL': 'equal'
}

rel2rel_simpler = {
    'is_included': 'during',
    'simultaneous': 'during',
    'before': 'before',
    'identity': 'during',
    'during': 'during',
    'ended_by': 'during',
    'begun_by': 'during',
    'iafter': 'after',
    'after': 'after',
    'ibefore': 'before',
    'ends': 'during',
    'includes': 'during',
    'during_inv': 'during',
    'begins': 'during',
    'none': 'none',
    'modal': 'modal',
    'evidential': 'evidential',
    'factual': 'factual',
    'equal': 'during',
    "vague": "vague",
    "counter_factive": "counter_factive",
    'factive': 'factive',
    'neg_evidential': 'neg_evidential',
    'conditional': 'conditional',
}

rel2opposite = {
    'during': 'during',
    'after': 'before',
    'before': 'after',
    'evidential': 'evidential',
    'modal': 'modal',
    'factual': 'factual',
    'counter_factive': 'counter_factive',
    'factive': 'factive',
    'neg_evidential': 'neg_evidential',
    'conditional': 'conditional',
    'vague': 'vague'
}

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

noun_counters = defaultdict(int)


class TimeBankStatistics:

    def print_counts(self):
        tmdr = TE3SilverDatareader()
        self._lf = LinguisticFeaturizer()
        self._matres_data_reader = MatresDataReader()
        timebank_documents = tmdr.timebank_documents('train')
        self._instance = 0
        self._tense_count = defaultdict(int)
        self._matres_dict = self._matres_data_reader.matres_dict('timebank')

        for documenti, document in enumerate(timebank_documents):
            print(document.file_name())
            for tlink in document.tlinks():
                if 't' in tlink.tlink_type():
                    continue
                tlink_type = tlink.tlink_type()
                if tlink_type[0] == 'e':
                    from_event = tlink.event_instance_id()

                if tlink_type[1:] == '2e':
                    to_event = tlink.related_to_event_instance()
                rel_type = tlink.rel_type()
                self._process_link(document, from_event, to_event, rel_type)

                    
            for slink in document.slinks():
                from_event = slink.event_instance_id()
                to_event = slink.subordinated_event_instance()
                rel_type = slink.rel_type()
                self._process_link(document, from_event, to_event, rel_type)

        for i in sorted(self._tense_count.items(), key=lambda x: (x[0][0], x[1])):
            print(i)


    def _process_link(self, document, from_event, to_event, rel_type):

        matres_key = (document.file_name(), from_event.strip('ei'), to_event.strip('ei'))
        matres_key_opp = (document.file_name(), to_event.strip('ei'), from_event.strip('ei'))

        if rel_type in ['EVIDENTIAL', 'MODAL', 'FACTUAL']:
            if matres_key in self._matres_dict:
                rel_type = self._matres_dict[matres_key][-1]
            elif matres_key_opp in self._matres_dict:
                rel_type = self._matres_dict[matres_key_opp][-1]

        from_sentence, from_sentence_i, from_token_i, from_start_token_i, from_end_token_i, from_token, from_token_global_i = self.event_instance_id2sentence(
            document, from_event, 'from',
        )

        to_sentence, to_sentence_i, to_token_i, to_start_token_i, to_end_token_i, to_token, to_token_global_i = self.event_instance_id2sentence(
            document, to_event, 'to',
        )


        token_order = self.token_order(
            from_sentence_i, from_token_i, to_sentence_i,
            from_sentence_i
        )
        if len(to_sentence) == 0 or len(from_sentence) == 0 or from_sentence != to_sentence:
            return

        lf1 = self._lf.featurize_sentence(' '.join(from_sentence))
        tokens = lf1.tokens()
        to_token_spacy = None
        from_token_spacy = None
        for token in tokens:
            if token.text() == from_token:
                from_token_spacy = token
            if token.text() == to_token:
                to_token_spacy = token
        
        
        if from_token_spacy is not None and from_token_spacy.text() in said_verbs:
            use = False
            parent = to_token_spacy
            seen = False
            while parent is not None:
                if parent.dep() == 'relcl':
                    seen = True
                if from_token_spacy.i() == parent.i() and seen is True:
                    use = True
                parent = parent.parent()
            from_tense, from_aspect = self.token2tense(from_sentence, from_token_spacy)
            to_tense, to_aspect = self.token2tense(from_sentence, to_token_spacy)
            rel_type = rel2rel_simpler[rel_type.lower()]
            tense = ('{}_{}_{}_{}'.format(str(from_tense), str(from_aspect), str(to_tense), str(to_aspect)), str(rel_type))
            
            if use is True and to_token_spacy is not None and from_token_spacy is not None:
                if tense == 'Past_None_Past_None_before':
                    print(' '.join(to_sentence))
                    print(from_token, to_token, from_tense, from_aspect, to_tense, to_aspect, rel_type)
                self._instance += 1
                self._tense_count[tense] += 1

        
        if to_token_spacy is not None and to_token_spacy.text() in said_verbs:
            use = False
            seen = False
            parent = from_token_spacy
            while parent is not None:
                if parent.dep() == 'ccomp':
                    seen = True
                if to_token_spacy.i() == parent.i() and seen is True:
                    use = True
                parent = parent.parent()
            from_tense, from_aspect = self.token2tense(from_sentence, from_token_spacy)
            to_tense, to_aspect = self.token2tense(from_sentence, to_token_spacy)
            rel_type = rel2rel_simpler[rel_type.lower()]
            rel_type = rel2opposite[rel_type]
            tense = ('{}_{}_{}_{}'.format(str(to_tense), str(to_aspect), str(from_tense), str(from_aspect)), str(rel_type))
            if use is True and to_token_spacy is not None and from_token_spacy is not None:
                #print(from_token, to_token, from_tense, from_aspect, to_tense, to_aspect, rel_type)
                # print(' '.join(to_sentence))
                #print()
                self._instance += 1
                self._tense_count[tense] += 1


    def event_instance_id2sentence(self, document, eiid, event_point):
        lf = LinguisticFeaturizer()
        make_instance = document.event_instance_id2make_instance(eiid)
        eid = make_instance.event_id()
        sseq = []
        sentence_i = None
        token_i = None
        start_token_i = None
        end_token_i = None
        token_text = None
        token_global_i = None
        if event_point == 'from':
            tags = ('[ENTITY_1]', '[/ENTITY_1]')
        elif event_point == 'to':
            tags = ('[ENTITY_2]', '[/ENTITY_2]')
        if document.is_eid(eid) is True:
            from_sentence = document.eid2sentence(eid)
            sentence_i = from_sentence.sentence_i()
            for s in from_sentence.sequence():
                if isinstance(s, TimebankEvent):
                    if s.eid() == eid:
                        #sseq.extend([tags[0], s.text(), tags[1]])
                        sseq.append(s.text())
                        token_i = s.sentence_token_i()
                        start_token_i = s.start_token_i()
                        end_token_i = s.end_token_i()
                        token_text = s.text()
                        token_global_i = from_sentence.sentence_start_token_global_i() + start_token_i
                    else:
                        sseq.append(s.text())
                else:
                    sseq.append(s.text())
        return sseq, sentence_i, token_i, start_token_i, end_token_i, token_text, token_global_i

    def token_order(self, from_sentence_i, from_token_i, to_sentence_i,
                    to_token_i):
        if from_sentence_i < to_sentence_i:
            return 0
        elif from_sentence_i > to_sentence_i:
            return 1
        else:
            if from_token_i < to_token_i:
                return 0
            else:
                return 1


    def token2tense(self, sentence, token):
        context = ' '.join(sentence)
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
                        if child.text() in pres_perf_aux:
                            tense = 'Pres'
            if aux_there is False and aspect == 'Perf':
                aspect = None
        
            if any(future_modal in context[max(0, token.idx() - 20): token.idx()].lower() for future_modal in future_modals):
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
    tbs = TimeBankStatistics()
    tbs.print_counts()