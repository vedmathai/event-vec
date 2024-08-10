from collections import defaultdict
import numpy as np

from eventvec.server.data.alphanli.alphanli_reader import AlphaNLIDataReader


other_conjuctions = {'and', 'because', 'but', 'so'}


class ConjunctionMutualInformation():
    
    def calculate(self):
        self._alphanli_datareader = AlphaNLIDataReader()
        data = self._alphanli_datareader.read_file('train')
        n_total, n_sentences, all_words = self.calculate_n_sentences(data)

        # Store all features in a list sorted in descending order by their mutual information.
        mi_list_unigrams = sorted([(self.mutual_information(w, n_total, n_sentences), w) for w in all_words], key=lambda x: x[0], reverse=True)
        for mi, w in mi_list_unigrams[:60]:
            contri = {}
            for category in other_conjuctions:
                contri[category] = n_sentences[category][w]
            print(sorted(contri.items(), key=lambda x: x[1], reverse=True), w, mi)


    def calculate_n_sentences(self, data):
        n_total = defaultdict(int)
        n_sentences = defaultdict(lambda: defaultdict(int))
        all_words = set()
        seen = set()
        for datum in data.data():
            if datum.label() == '1':
                passage = f"{datum.obs_1()} {datum.hyp_1()}."
            else:
                passage = f"{datum.obs_1()} {datum.hyp_2()}."
            key = passage + ' ' + datum.obs_2()
            if key in seen:
                continue
            seen.add(key)
            word_split= key.split()
            key_bigrams = ['{} {}'.format(word_split[i], word_split[j]) for i in range(len(word_split) - 1) for j in range(i + 1, len(word_split))]
            word_cloud = set(key_bigrams)
            all_words |= word_cloud
            for connector in other_conjuctions:
                if connector in key.split():
                    n_total[connector] += 1
                    for context_word in list(word_cloud):
                        if connector not in context_word:
                            n_sentences[connector][context_word] += 1
        return n_total, n_sentences, all_words


    def mutual_information(self, word, n_total, n_sentences):
        # Find the counts for the word in each combination of hateful/non-hateful and in/out of the tweet
        # Write your code here (Fill in the right hand side of the following) 
        in_counts = defaultdict(int)
        out_counts = defaultdict(int)
        for connector in other_conjuctions:
            in_counts[connector] = n_sentences[connector][word]
            out_counts[connector] = n_total[connector] - n_sentences[connector][word]

        # Find the total number of tweets
        # Write your code here (Fill in the right hand side of the following)
        N = sum(n_total.values())

        # Calculate the probabilities
        # Write your code here (Fill in the right hand side of the following)
        in_prob_dict = {key: in_counts[key] / N for key in in_counts}
        out_prob_dict = {key: out_counts[key] / N for key in out_counts}

        in_prob = sum(in_prob_dict.values())
        out_prob = sum(out_prob_dict.values())

        # Calculate the marginal probabilities
        # Write your code here (Fill in the right hand side of the following)


        # Define function to calculate mutual information for a given word take care for 0 probabilities
        def mi_formula(pxy, px, py):
            # Write your code here
            if px * py == 0 or pxy == 0:
                return 0
            else:
                return pxy * np.log2(pxy / (px * py))
        
        # Calculate mutual information for each combination of hateful/non-hateful and in/out of the tweet
        # Write your code here (Fill in the right hand side of the following)
        total_mi = 0
        for connector in other_conjuctions:
            marginal = in_prob_dict[connector] + out_prob_dict[connector]

            mi_in = mi_formula(in_prob_dict[connector], marginal, in_prob)
            mi_out = mi_formula(out_prob_dict[connector], marginal, out_prob)
            total_mi += mi_in + mi_out

        
        return total_mi


if __name__ == '__main__':
    cmi = ConjunctionMutualInformation()
    cmi.calculate()
