from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_document import TimebankDocument  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_timex import TimebankTimex  # noqa

from eventvec.server.data_readers.timebank_reader.timebank_dense_reader import TimeBankDenseDataReader  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_event import TimebankEvent  # noqa

from collections import defaultdict

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
}

rel2rel_simpler = {
    'IS_INCLUDED': 'during',
    'SIMULTANEOUS': 'during',
    'BEFORE': 'before',
    'IDENTITY': 'during',
    'DURING': 'during',
    'ENDED_BY': 'during',
    'BEGUN_BY': 'during',
    'IAFTER': 'after',
    'AFTER': 'after',
    'IBEFORE': 'before',
    'ENDS': 'during',
    'INCLUDES': 'during',
    'DURING_INV': 'during',
    'BEGINS': 'during',
    'NONE': 'none',
}

noun_counters = defaultdict(int)


class TimeBankStatistics:

    def print_counts(self):
        tmdr = TimeBankDenseDataReader()
        timebank_documents = tmdr.timebank_documents('train')
        data = []
        counter_pos = defaultdict(int)
        counter_same_pos = defaultdict(int)
        rel_counter = defaultdict(int)
        rel_counter_simpler = defaultdict(int)
        token_order_counter = {0: 0, 1: 0}

        for document in timebank_documents:
            print(document.eids())
            for tlink in document.tlinks():
                pos = ''
                if 't' in tlink.tlink_type():
                    continue
                tlink_type = tlink.tlink_type()
                if tlink_type[0] == 'e':
                    from_event = tlink.event_instance_id()
                    make_instance = document.event_instance_id2make_instance(from_event)
                    pos = make_instance.pos()
                    from_pos = make_instance.pos()
                    from_sentence, from_sentence_i, from_token_i, from_start_token_i, from_end_token_i, from_token = self.event_instance_id2sentence(
                        document, from_event, 'from',
                    )
                if tlink_type[1:] == '2e':
                    to_event = tlink.related_to_event_instance()
                    make_instance = document.event_instance_id2make_instance(to_event)
                    pos += '|' + make_instance.pos()
                    to_pos = make_instance.pos()
                    to_sentence, to_sentence_i, to_token_i, to_start_token_i, to_end_token_i, to_token = self.event_instance_id2sentence(
                        document, to_event, 'to',
                    )

                token_order = self.token_order(
                    from_sentence_i, from_token_i, to_sentence_i,
                    from_sentence_i
                )
                if pos in ['VERB|NOUN']:
                    token_order_counter[token_order] += 1

                if len(to_sentence) == 0 or len(from_sentence) == 0:
                    continue

                if pos == 'NOUN|NOUN':
                    if from_token is not None and to_token is not None:
                        noun_pairs = '|'.join([from_token.lower(), to_token.lower()])
                        noun_counters[noun_pairs] += 1
                        print(from_token, to_token)

                counter_pos[pos] += 1

                if to_sentence_i == from_sentence_i:
                    counter_same_pos[pos] += 1

                rel_counter[rel2rel_simpler[tlink.rel_type()]] += 1

        for k, i in sorted(counter_pos.items(), key=lambda x: x[1], reverse=True):
            print(k, i)
        print('\n')
        for i in counter_same_pos:
            print(i, counter_same_pos[i])
        print('\n')
        for rel_type in rel_counter:
            print(rel_type, rel_counter[rel_type])
        print('noun_pairs', noun_counters, sum(noun_counters.values()))
        print(token_order_counter)

    def event_instance_id2sentence(self, document, eiid, event_point):
        make_instance = document.event_instance_id2make_instance(eiid)
        eid = make_instance.event_id()
        sseq = []
        sentence_i = None
        token_i = None
        start_token_i = None
        end_token_i = None
        token_text = None
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
                        sseq.extend([tags[0], s.text(), tags[1]])
                        token_i = s.sentence_token_i()
                        start_token_i = s.start_token_i()
                        end_token_i = s.end_token_i()
                        token_text = s.text()
                    else:
                        sseq.append(s.text())
                else:
                    sseq.append(s.text())
        return sseq, sentence_i, token_i, start_token_i, end_token_i, token_text

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


if __name__ == '__main__':
    tbs = TimeBankStatistics()
    tbs.print_counts()