import pprint
from matplotlib import pyplot as plt
from collections import defaultdict
from scipy.spatial import distance as cosine_distance
import numpy as np
import scipy.stats as stats
from transformers import RobertaTokenizer, RobertaModel


from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_document import TimebankDocument  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_timex import TimebankTimex  # noqa

from eventvec.server.data_readers.timebank_reader.timebank_dense_reader import TimeBankDenseDataReader  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_event import TimebankEvent  # noqa



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
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        tmdr = TimeBankDenseDataReader()
        timebank_documents = tmdr.timebank_documents('train')
        distance_counter = defaultdict(lambda: defaultdict(int))
        data = []
        counter_pos = defaultdict(int)
        counter_same_pos = defaultdict(int)
        rel_counter = defaultdict(int)
        rel_counter_simpler = defaultdict(int)
        token_order_counter = {0: 0, 1: 0}
        cosine_distances = defaultdict(list)
        words = defaultdict(set)
        will_counter = 0
        self._cache = {}

        for documenti, document in enumerate(timebank_documents):
            print(documenti)
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
                    from_sentence, from_sentence_i, from_token_i, from_start_token_i, from_end_token_i, from_token, from_token_global_i = self.event_instance_id2sentence(
                        document, from_event, 'from',
                    )
                if tlink_type[1:] == '2e':
                    to_event = tlink.related_to_event_instance()
                    make_instance = document.event_instance_id2make_instance(to_event)
                    pos += '|' + make_instance.pos()
                    to_pos = make_instance.pos()
                    to_sentence, to_sentence_i, to_token_i, to_start_token_i, to_end_token_i, to_token, to_token_global_i = self.event_instance_id2sentence(
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

                counter_pos[pos] += 1

                if to_sentence_i == from_sentence_i:
                    counter_same_pos[pos] += 1

                rel_counter[rel2rel_simpler[tlink.rel_type()]] += 1
                distance = abs(from_token_global_i - to_token_global_i)

                rel = rel2rel_simpler[tlink.rel_type()]
                if pos in ['NOUN|NOUN', 'VERB|VERB'] and rel in ['after', 'before', 'during']:
                    d = self.cosine_sim(from_token, to_token)
                    key = '{}-{}'.format(pos, rel)
                    cosine_distances[key].append(d)
                    distance_counter[key][distance] += 1
                words[from_pos].add(from_token)
                words[to_pos].add(to_token)

        for pos in ['VERB|VERB', 'NOUN|NOUN']:
            from_pos, to_pos = pos.split('|')
            for token1 in list(words[from_pos]):
                for token2 in list(words[to_pos]):
                    d = self.cosine_sim(token1, token2)
                    cosine_distances[pos].append(d)

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
        handles = []
        for rel in sorted(distance_counter.keys()):
            array = [0 for i in range(50)]
            for ii, i in enumerate(array):
                if ii in distance_counter[rel]:
                    array[ii] = distance_counter[rel][ii]
            handle, = plt.plot([np.mean(array[i: i+5]) for i in range(len(array)-5)], label=rel, linewidth=3)
            handles.append(handle)
        plt.legend(handles=handles)
        plt.savefig('/home/lalady6977/Pictures/histogram-unbucketed')
        plt.clf()
        handles = []
        for key, value in sorted(cosine_distances.items(), key=lambda x: x[0]):
            mean = np.mean(value)
            std = np.std(value)
            print(key, 'std', std, 'mean', mean)
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            handle, = plt.plot(x, stats.norm.pdf(x, mean, std), label=key, linewidth=2)
            handles.append(handle)
        plt.legend(handles=handles)
        plt.savefig('/home/lalady6977/Pictures/normal-distribution-cosine-similarities')

    def event_instance_id2sentence(self, document, eiid, event_point):
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
                        sseq.extend([tags[0], s.text(), tags[1]])
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

    def cosine_sim(self, from_token, to_token):
        lf = LinguisticFeaturizer()
        if from_token not in self._cache:
            lf1 = lf.featurize_sentence(from_token)
            token1_vector = np.mean([i.vector() for i in lf1.tokens()], axis=0)
            """
            token1_input_ids = self.tokenizer(
                [from_token],
                padding='max_length',
                max_length=100,
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=True
            )['input_ids']
            hidden_output1, token1_vector = self.roberta(
                input_ids=token1_input_ids,
                return_dict=False
            )
            self._cache[from_token] = hidden_output1.detach().numpy()[0][1]
            """
            self._cache[from_token] = token1_vector
        vector1 = self._cache[from_token]

        if to_token not in self._cache:
            lf2 = lf.featurize_sentence(to_token)
            token2_vector = np.mean([i.vector() for i in lf2.tokens()], axis=0)
            """
            token2_input_ids = self.tokenizer(
                [to_token],
                padding='max_length',
                max_length=5,
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=True
            )['input_ids']

            hidden_output2, token2_vector = self.roberta(
                input_ids=token2_input_ids,
                return_dict=False
            )
            self._cache[to_token] = hidden_output2.detach().numpy()[0][1]
            """
            self._cache[to_token] = token2_vector
        vector2 = self._cache[to_token]
        d = cosine_distance.cosine(vector1, vector2)
        # d = cosine_distance.cosine(token1_vector, token2_vector)
        return d



if __name__ == '__main__':
    tbs = TimeBankStatistics()
    tbs.print_counts()